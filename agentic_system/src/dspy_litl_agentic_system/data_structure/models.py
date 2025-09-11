"""
models.py

Pydantic models for structured input/output for agentic prediction task.
Itended to allow for easier tracking of agentic system reasoning to facilitate
debugging, downstream evaluation, and creation of agentic memory system. 

Classes:
- LitlIdentifier: Minimal drug identifier and experimental context
- LitlInput: A single prediction task unit
- LitlPrediction: Structured prediction output from the agent
- LitlResponseUnit: Full result for one input unit, ready to persist
"""

from __future__ import annotations
from typing import Literal, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field

IdentifierType = Literal["name", "prism_id", "smiles"]

class LitlIdentifier(BaseModel):
    """
    Minimal handle the agent can start from; 
    This makes only the drug identifier, experimental description, 
    and the truth ic50 value (during agent reflection phase) 
    accessible to the agentic system.

    All other drug specific context should be fetched via tools.
    """
    type: IdentifierType
    value: str

    # should also include cell line information if applicable
    experimental_description: str

class LitlInput(BaseModel):
    """
    A single LITL task unit: one drug to predict. 
    Keep it tiny & uniquely identifiable.
    """
    unit_id: str = Field(default_factory=lambda: f"unit-{uuid4().hex[:12]}")
    identifier: LitlIdentifier
    output_unit: str

class LitlPrediction(BaseModel):
    """
    Structured prediction output from the agent for one input unit.
    """
    value: str # the same as identifier.value
    predicted_ic50: float
    confidence: int
    explanation: str
    trajectory: Dict[str, Any] # full LLM reasoning trajectory

class LitlUnit(BaseModel):
    """
    Full representation of one LITL prediction. 
    """
    unit_id: str
    iter: int
    identifier: LitlIdentifier
    prediction: LitlPrediction
