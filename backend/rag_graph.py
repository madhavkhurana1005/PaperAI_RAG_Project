# JAI SHREE RAMCHANDRA JI KI JAI

import os
import sqlite3 
import warnings
from typing import Annotated

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool,InjectedToolCallId
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langgraph.types import Command

from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import RouterDecision,RelevancyDecision, ClaimVerificationResult
from backend.vector_store import search as vs_search

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")


#-----------------------------STATE------------------------------------------------------------------------

class RAGState(MessagesState):
    session_id : str
    query : str
    route : str | None
    retrieved_docs : list[Document]
    retrieved_attempts : int
    claim_verdict : str | None
    claim_source : str | None
    superseding_papers : list[dict] | None
    answer : str | None
    is_relevant : bool | None
    rewrite_count : int | None



#-----------------------------ROUTER------------------------------------------------------------------------

ROUTE_PROMPT = ChatPromptTemplate.from_messages([("system",
"You are a routing assistant for a research papaer Question and Answer Application.  \n "
"You need to route user query into only categories. i.e. retrieve, verify_claim, direct_answer \n"
"choose retrieve when you need information from the research paper or when you need latest information  for (eg. checking weather of delhi) \n"
"The timesensitive information which is not present in your training data, use retrieve \n"
"choose verify_claim when user explicitly wants to verify some claim/ source from the research fields. \n"
"choose direct_answer for normal queries where user does not want to verify any claim, neither need latest information nor information that is present in your training data. \n"
"direct_answer queries like what is capital of a country or definition of a particular topic or reasoning queries that can be easily solved by you without external knowledge \n"
"when in confusion between direct_answer or retrieve, always choose  retrieve."),
("human","{query}")
]
)

router_chain = ROUTE_PROMPT | llm.with_structured_output(RouterDecision)

def router_node(state : RAGState) -> dict:
    query = state["messages"][-1].content
    decision : RouterDecision = router_chain.invoke({"query":query})
    return {"route": decision.route}


#-----------------------------TOOL SCHEMA----------------------------------------------------------------------

class RetrieverInput(BaseModel):
    query : str = Field(description="semantic query to search research paper chunks")
    k : int = Field(default=4, ge=1, le=10, description="nunber of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query : str = Field(description="rewritten and optimized query for web search")
    max_results : int = Field(default=4, ge=1, le=10, description="no of results to fetch during web search")



#-----------------------------TOOLS----------------------------------------------------------------------

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query : str, k : int,
    session_id : Annotated[str ,InjectedState("session_id")],
    current_docs : Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id : Annotated[str, InjectedToolCallId]
) -> Command:
    """Search the uploaded research paper vector store for relevant passage"""
    docs = vs_search(query=query,session_id=session_id, k=k)
    if not docs:
        return Command(update={"messages": [ToolMessage(content = "No documents retrieved", tool_call_id = tool_call_id)]})
    tool_retrive_summary = f"Total {len(docs)} documents retrieved"

    return Command(update={
        "messages": [ToolMessage(content = tool_retrive_summary, tool_call_id = tool_call_id)],
        "retrieved_docs": (current_docs or []) + docs,
    })

@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query : str,
    max_results : int,
    current_docs : Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id : Annotated[str, InjectedToolCallId]
) -> Command:
    """Search the web for relevant information using Tavily"""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    docs = tavily.search(query=optimized_query,max_results=max_results)

    if not docs.get("results"):
        return Command(update={"messages": [ToolMessage(content="No web results fetched", tool_call_id = tool_call_id)]})

    web_docs = [ Document(page_content=d["content"], metadata = {"url" : d["url"], "title" : d.get("title", "Web Result")})
                 for d in docs["results"] ]

    web_fetch_summary = f"Total {len(web_docs)} documents fetched from web"

    return Command(update={
        "messages": [ToolMessage(content=web_fetch_summary, tool_call_id = tool_call_id)],
        "retrieved_docs": (current_docs or []) + web_docs,
    })



#-----------------------------RETRIVAL AGENT----------------------------------------------------------------------

retrieval_tools = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(retrieval_tools,parallel_tool_calls=False)
base_tool_node = ToolNode(retrieval_tools)

RETRIEVAL_SYSTEM = (
"You are a research assistant gathering context to answer a user's question about research papers.\n\n"
"You have an user query as an input you have to decide from 2 tools aviailable to you, which one to call inorder to properly answer user query\n"
"1. retrieve_from_vectorstore :  use this tool when user query is related to uploaded research papers, fetch the chunks simialr to user query, perform semantic search\n"
"You have full authority to change any parameter of tool function in order to retriev better results and answer user query in a better way.\n"
"You can change the value of k (number of chunks to retrieve) in order to make the answer exhaustive, relevant and useful for user.\n"
"2. web_search: use this tool when retrived documents are not sufficient to answer the user query properly. Or if you need additional information to support the answer or complete the answer\n"
"web_search tool should be only used for answering the queries that requires data not present in your training data, for example latest or time sensitive information etc. \n"
"You can change the value of max_results (number of results to fetch from web) in order to make the answer exhaustive, relevant and useful for user.\n"
"Call only one tool per turn, do not product answer for user, just make the tool call and collect the context.\n"
)


RELEVANCY_CHECK_SYSTEM = (
"You are evaluating assistant, evaluating whether the retrieved context is relevant enough to answer the user query or not \n"
"Return true if the retrieved context is sufficient to answer user query, even partially \n"
"Return false if the retrieved context is not sufficient to answer user query. Return false if retrieved context chunks seems off topic or irrelevant from the user query. \n"
"Be lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)


QUERY_REWRITE_SYSTEM = (
"You are a query rewriting assistant. The previous user query was failed to fetch the relevant chunks from vector store or web results"
"to answer user question properly. rewrite the query in such a way that better chunks are retrieved from vector store or from web search results."
"The goal is to simplify the query, rephrase it in such a way that semantic search can be done properly. Return only the rewritten query, nothing else."
)

#-----------------------------NODES----------------------------------------------------------------------


def agent_node(state : RAGState) -> dict:
    current_attempts = state.get("retrieved_attempts",0)
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    messages = [{"role": "system", "content":RETRIEVAL_SYSTEM}] + state["messages"]
    response = lm.invoke(messages)
    updates : dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieved_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs") or []

    data = "----".join(i.page_content[:500] for i in retrieved_docs[:3])
    if not data:
        return {"is_relevant" : False}
    user_prompt = (f"user query : {query}, retrieved documents : {data}\n"
              "Are these retrieved documents relevant to answer my question?")

    messages = [
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    decision : RelevancyDecision = relevancy_llm.invoke(messages)
    return {"is_relevant": decision.is_relevant}

def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count",0)
    user_prompt = (f"original query : {original_query} \n"
              "Rewrite this query to fetch better results")

    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)
    rewritten_query = response.content.strip()


    return {"messages": [HumanMessage(content=rewritten_query)],
            "is_relevant" : None,
            "rewrite_count" : rewrite_count + 1,
            "query" : rewritten_query,
            "retrieved_docs" : [],
            "retrieved_attempts": 0,
            }

CLAIM_VERIFICATION_SYSTEM = (
    "You are a research fact checker assistant, Given a claim from research paper and a set of recent web and arxiv results, determine \n"
    "1. Has the claim been superseded, significantly challenged, or updated by more recent work? \n"            
    "2. Identify upto 3 research papers that proves that the clain is superseded or challenged \n"
    "Rules : \n"
    "Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
    )

verification_llm = llm.with_structured_output(ClaimVerificationResult)

def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # GENERAL WEB SEARCH
    general_results = tavily_client.search(f"recent research superseding : {claim[:500]}", max_results=5).get("results",[])

    # ARXIV SEARCH
    arxiv_results = tavily_client.search(f"site:arxiv.org {claim[:500]}", max_results=5).get("results",[])

    # lines = ["====GENERAL WEB SEARCH RESULT======"]
    # for r in general_results:
    #     lines.append(
    #         f"Title: {r.get("title"," ")}\n"
    #         f"URL: {r.get("url"," ")}\n"
    #         f"Snippet: {r.get("content"," ")[:300]}\n")
    # lines = ["====ARXIV SEARCH RESULT======"]
    # for r in arxiv_results:
    #     lines.append(
    #         f"Title: {r.get("title"," ")}\n"
    #         f"URL: {r.get("url"," ")}\n"
    #         f"Snippet: {r.get("content"," ")[:300]}\n")     
        
    lines = ["====GENERAL WEB SEARCH RESULT======"]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', ' ')}\n"
            f"URL: {r.get('url', ' ')}\n"
            f"Snippet: {r.get('content', ' ')[:300]}\n"
        )

    lines.append("====ARXIV SEARCH RESULT======")

    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', ' ')}\n"
            f"URL: {r.get('url', ' ')}\n"
            f"Snippet: {r.get('content', ' ')[:300]}\n"
        )
            
    context = "\n".join(lines)


    user_prompt = (f"original claim : {claim} \n"
                f"search results : {context}")

    messages = [
        {"role": "system", "content": CLAIM_VERIFICATION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    response : ClaimVerificationResult = verification_llm.invoke(messages)

    papers_dicts = [r.model_dump() for r in response.supeseded_docs[:3]]  #converts pydantic object into dict

    return {
        "claim_verdict" : response.verdict_summary,
        "superseding_papers" : papers_dicts,
        "claim_source" : papers_dicts[0]["url"] if papers_dicts else None
    }
    
def generate_answer_node(state:RAGState) -> dict:
    route = state.get("route")
    query = state.get("query")

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count",0) >= 1:
            answer = "I am not able to get relevant information for your query. Try rephrasing your question or uploading more documents."
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer for your query"
            else:
                context = "\n\n===\n".join(d.page_content for d in docs)

                prompt = f"Question : {query}, context : {context}. Answer the above question using the given context"
                answer = llm.invoke([{"role":"user", "content" :prompt }]).content

    elif route == "verify_claim":
        verdict = state.get("claim_verdict","")
        papers = state.get("superseding_papers","") or []
        claim_text = state.get("query")

        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                "Claim Verification Result\n"
                f"Claim : {claim_text}\n"
                f"Verdict : {verdict}\n"
                f"Superseding Papers : {papers_block}\n"
                "You can refer any of these research papers to continue your research findings."
            )
        else: 
            answer = (
                "Claim Verification Result\n"
                f"Claim : {claim_text}\n"
                f"Verdict : {verdict}\n"
                f"Superseding Papers : Not Found\n"
                "The claim is correct, could not find anything superseding it"
            )
    else:
        prompt = f"Answer from your knowledeg, question : {query}"
        answer = llm.invoke([{"role":"user","content": prompt}]).content

    return {"answer":answer,"messages":[AIMessage(content=answer)]}


#-----------------------------GRAPH----------------------------------------------------------------------


MAX_RETRIEVAL_ATTEMPTS = 3

def route_query(state:RAGState) -> str:
    return state["route"]

def agent_routing(state:RAGState) -> str:
    tc = tools_condition(state)
    if tc == "tools":            #It checks the last AI message.
        return "retrieval"
    if state.get("retrieved_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"

def after_relevancy_routing(state:RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count",0) < 1:
        return "rewrite_query"
    return "generate_answer"

def build_graph(db_path : str = "checkpoints.db"):
    connection = sqlite3.connect(database = db_path, check_same_thread=False) # persist graph state in db
    checkpointer = SqliteSaver(connection)

    graph = StateGraph(RAGState)
    graph.add_node("router",router_node)
    graph.add_node("agent_node",agent_node)
    graph.add_node("retrieval",base_tool_node)
    graph.add_node("relevancy_check",relevancy_check_node)
    graph.add_node("query_rewrite",query_rewrite_node)
    graph.add_node("verify_claim",verify_claim_node)
    graph.add_node("generate_answer",generate_answer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router",route_query,{"retrieve": "agent_node", 
                                                      "verify_claim":"verify_claim",
                                                       "direct_answer":"generate_answer"})

    graph.add_conditional_edges("agent_node",agent_routing,{"retrieval": "retrieval", 
                                                      "generate_answer":"generate_answer",
                                                       "relevancy_check":"relevancy_check"})
    
    graph.add_edge("retrieval","agent_node")

    graph.add_conditional_edges("relevancy_check",after_relevancy_routing,{"rewrite_query": "query_rewrite", 
                                                      "generate_answer":"generate_answer"})
    
    graph.add_edge("query_rewrite", "agent_node")
    graph.add_edge("verify_claim","generate_answer")
    graph.add_edge("generate_answer",END)

    return graph.compile(checkpointer=checkpointer)