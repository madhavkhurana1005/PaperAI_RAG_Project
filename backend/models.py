from typing import Literal
from pydantic import BaseModel

class RouterDecision(BaseModel):
    route : Literal["retrieve","direct_answer","verify_claim"]

class RelevancyDecision(BaseModel):
    is_relevant : bool
    reason : str

class SupersedingPaper(BaseModel):
    title : str
    url : str
    summary : str

class ClaimVerificationResult(BaseModel):
    is_superseded : bool
    verdict_summary : str
    supeseded_docs : list[SupersedingPaper]

class BtwRouteDecision(BaseModel):
    needs_web_search : bool