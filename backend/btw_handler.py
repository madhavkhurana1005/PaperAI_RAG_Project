import os 
from typing import Generator
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from backend.models import BtwRouteDecision

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")

def handle_btw(query : str) -> Generator[str,None,None]:
    """ Off topic side channel, never touches the main checkpointer or vector store"""
    route_prompt = ChatPromptTemplate.from_messages([("system","Decide if answering this question requires a real-time web search (recent events,current prices, breaking news) or if your general knowledge is sufficient."),("human","{query}")])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query":query})

    if decision.needs_web_search:
        tavily_search = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = tavily_search.search(query=query,max_results=3)
        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

        prompt = ChatPromptTemplate.from_messages([("system","Answer the user query from web search results. Results: {context}, sources : {sources}"),("human","{query}")])
        stream = (prompt | llm).stream({"query": query, "context": context, "sources": sources})

    else:

        prompt = ChatPromptTemplate.from_messages([("system","Answer the user query from self knowledge."),("human","{query}")])
        stream = (prompt | llm).stream({"query": query})

    for chunk in stream:
        if chunk.content:
            yield chunk.content
