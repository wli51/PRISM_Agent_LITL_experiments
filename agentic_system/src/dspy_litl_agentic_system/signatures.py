"""
signatures.py

Collection of dspy.Signature classes for the LITL IC50 prediction task.
A DSPy signature declaratively defines the behavior of an agent via defining
input and output fields. 
For simplicity, currently only one signature is defined predicting from
a single drug identifier and relevant experimental context.

Classes:
- PredictIC50FromIdentifier: Predict IC50 from a single drug identifier
"""

import dspy

class PredictIC50FromIdentifier(dspy.Signature):
    """
    You are a expert pharmacologist and medicinal chemist, tasked with
    predicting the IC50 of a drug against a specific biological target.
    You are given a single drug identifier (name, PRISM ID, or SMILES).
    If available, additional tools can be used to look up more information 
    about the drug target, mechanism of action, etc. If you would like to 
    acquire such information, you MUST explicitly call these tools.
    """
    # for some flexibility to use different kinds of drug identifiers
    identifier_type: str = dspy.InputField(
        desc="One of: name | prism_id | smiles")
    identifier_value: str = dspy.InputField(
        desc="The actual identifier string")
    experimental_description: str = dspy.InputField(
        desc="A brief description of the biological target or assay context "
             "(e.g. cell line, protein target, etc.) for which you are "
             "predicting the IC50")
    
    drug_identifier_out: str = dspy.OutputField(
        desc="The input drug identifier repeated back for clarity")
    predicted_ic50: float = dspy.OutputField(
        desc="Your predicted IC50 value (in nM) for the drug against the target")
    confidence: int = dspy.OutputField(
        desc="Your confidence in the IC50 prediction, on a scale of 0-100")
    explanation: str = dspy.OutputField(
        desc="A brief explanation of how you arrived at your IC50 prediction")
