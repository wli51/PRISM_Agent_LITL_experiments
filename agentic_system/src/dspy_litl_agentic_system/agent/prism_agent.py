# prism_agent.py

import dspy

from ..signatures import PredictIC50FromIdentifier

def get_basic_agent() -> dspy.ReAct:
    return dspy.ReAct(
        PredictIC50FromIdentifier,
        tools = []
    )