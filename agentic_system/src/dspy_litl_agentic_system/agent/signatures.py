"""
signatures.py

Collection of dspy.Signature classes for the LITL IC50 prediction task.
A DSPy signature declaratively defines the behavior of an agent via defining
input and output fields. 
For simplicity, currently only one signature is defined predicting from
a single drug identifier and relevant experimental context.

Classes:
- PredictIC50DrugCell: Predicts the IC50 value for a given drug and cell line,
  along with confidence and explanation.
"""

from typing import Optional, List, Dict, Any

import dspy

class PredictIC50DrugCell(dspy.Signature):
    """
    You are an expert pharmacologist and medicinal chemist.

    You are given a single drug name that uniquely identifies the drug,
    a single cell line name that uniquely identifies the cell line,
    and optionally an experimental description that provides
    additional context about the assay. 

    You are tasked with the provisional predicting of the 
    not previously tested cell viability IC50 value for a given drug
    against a specific cell line, based on existing knowledge bases
    accessible to you as tools. Explicitly call these tools to gather
    relevant information to inform your prediction. 
    
    Optionally, tool and task specific higher level context may be specified. 
    If available, follow/reference these higher level instructions.    
    """
    # for some flexibility to use different kinds of drug identifiers
    drug: str = dspy.InputField(
        desc="The drug name or identifier for which you will predict the IC50")
    cell_line: str = dspy.InputField(
        desc="The cell line name or identifier against which you will "
             "predict the IC50")
    experimental_description: Optional[str] = dspy.InputField(
        desc="Optional description of experimental details that may be "
             "relevant for predicting the IC50, or None if not available")
    output_unit: str = dspy.InputField(
        desc="The unit required for the predicted IC50 value")
    
    tool_context: Optional[str] = dspy.InputField(
        desc="Optional higher level context on tool usage"
    )
    additional_bio_context: Optional[str] = dspy.InputField(
        desc="Optional additional context that might be useful for the prediction task"
    )

    ic50_pred: float = dspy.OutputField(
        desc="Your predicted IC50 value (in `output_unit` scale) for the drug "
             "against the target, must be a float value strictly greater than 0"
        )
    confidence: int = dspy.OutputField(
        desc="Your confidence in the IC50 prediction, on a scale of 0-100")
    explanation: str = dspy.OutputField(
        desc="A detailed explanation of how you arrived at your IC50 prediction"
    )


class ReflectIC50DrugCell(dspy.Signature):
    """
    You are a senior expert pharmacologist and medicinal chemist. 
    Your colleague has provided a provision prediction of the cell viability,
    as IC50 value, for a given drug against a specific cell line,
    from exclusively existing knowledge without actual experimentation.
    
    Now the laboratory has obtained the true experimental IC50 value.
    
    Your task is to critically evaluate this provisional IC50 prediction,
    against the true IC50 value. You will be given all details of your colleague's
    prediction process, including the drug and cell line identifiers,
    experimental description, predicted IC50 value, confidence, explanation,
    and optionally the detailed trajectory of their reasoning steps.
    You will also have access to all tools that your colleague had access to.
    The over-arching goal of you task is to identify sound and flawed reasoning
    steps and tool utilization and provide detailed, specific calibration notes
    to inform the future improvement of IC50 predictions for this drug.

    All of you calibration notes should be specific and strictly actionable with the
    available tools and knowledge base. Do NOT propose unrealistic actions such as
    "conducting new experiments", "consulting external experts", or "use additional tools".
    """
    drug: str = dspy.InputField(
        desc="The drug name or identifier of the provisional IC50 prediction")
    cell_line: str = dspy.InputField(
        desc="The cell line name or identifier of the provisional IC50 prediction")
    experimental_description: Optional[str] = dspy.InputField(
        desc="Optional description of experimental details that may be "
             "relevant for predicting the IC50, or None if not available")
    output_unit: str = dspy.InputField(
        desc="The unit the provisional IC50 prediction is in")
    ic50_true: float = dspy.InputField(
        desc="The true IC50 value for the drug against the cell line "
             "from experimental measurement, in the same unit as `output_unit`."
    )
    ic50_pred: float = dspy.InputField(
        desc="The provisional IC50 prediction for the drug against the cell line "
             "in the same unit as `output_unit`."
    )
    confidence: int = dspy.InputField(
        desc="The confidence (0-100) associated with the provisional IC50 prediction."
    )
    explanation: str = dspy.InputField(
        desc="The explanation provided for the provisional IC50 prediction."
    )
    provisional_trajectory: Optional[str] = dspy.InputField(
        desc="Optional detailed trajectory of the provisional IC50 prediction, "
             "if available."
    )
    existing_drug_calibration: Optional[str] = dspy.InputField(
        desc="Optional existing drug-specific calibration notes, if available."
    )
    existing_cell_line_calibration: Optional[str] = dspy.InputField(
        desc="Optional existing cell line-specific calibration notes, if available."
    )

    verdict: str = dspy.OutputField(
        desc="A concise verdict and summary of how the provisional IC50 prediction "
             " compares to the true IC50 value. Always explicitly include the " 
             "predicted and true IC50 values in the verdict."
    )
    drug_specific_calibration: str = dspy.OutputField(
        desc="A detailed drug-specific calibration of the provisional IC50 prediction. "
             "Analyze the reasoning steps taken, identify any flaws or gaps in knowledge, "
             "and provide specific suggestions on how to improve the IC50 prediction "
             "for this drug in future attempts."
             "If existing drug-specific calibration notes are available, "
             "make your best call to incorporate (summarize existing and add new) "
             "or discard existing notes completely."

    )
    cell_line_specific_calibration: str = dspy.OutputField(
        desc="A detailed cell line-specific calibration of the provisional IC50 prediction. "
             "Analyze the reasoning steps taken, identify any flaws or gaps in knowledge, "
             "and provide specific suggestions on how to improve the IC50 prediction "
             "for this cell line in future attempts."
             "If existing cell line-specific calibration notes are available, " \
             "make your best call to incorporate (summarize existing and add new) " \
             "or discard existing notes completely."
    )
    task_specific_calibration: str = dspy.OutputField(
        desc="A detailed calibration note specific to this exact drug-cell line IC50 prediction. "
             "Analyze the reasoning steps taken, identify any flaws or gaps in knowledge, "
             "and provide specific suggestions on how to improve the IC50 prediction "
             "for this drug-cell line pair in future attempts."
    )
    general_notes: List[str] = dspy.OutputField(
        desc="A list of general notes and takeaways that can help improve "
            "future IC50 predictions that generalize across different drugs and cell lines. "
            "Every single note should be concise and standalone information in natural language."
    )


class BatchReflect(dspy.Signature):
    """
    You are a senior expert pharmacologist and medicinal chemist. 
    Several of your colleagues have provided a provisional prediction of 
    cell viability IC50 value for a given drug against a specific cell line,
    from exclusively existing knowledge without experimentation.

    Now the laboratory has conducted an actual assay to measure the true IC50 value
    and returned you with the results.

    To enhance the reliability of future provisional predictions, it is your
    task to critically evaluate your colleagues' predictions and reasonings,
    comparing them against the experimentally determined IC50 value. Provide
    specific, concise and actionable instructions on future predictions,
    separately for:

        1. The specific drug (see detailed output description)

        2. The specific cell line (see detailed output description)

    Output each of the two sets of instructions as natural language most usable
    by your colleagues in future IC50 predictions. If available, 
    you will be provided pre-existing instruction sets from your previous evaluation,
    or 'None' to indicate no prior instructions exist. 
    
    Please note that your outputs will always be used to overwrite these prior instructions.
    Please use your best call to decide to improve upon, integrate into, 
    or completely overwrite prior instructions for the ultimate benefit of future predictions
    """

    drug: str = dspy.InputField(
        desc="The drug name or identifier for which you will predict the IC50")
    cell_line: str = dspy.InputField(
        desc="The cell line name or identifier against which you will "
             "predict the IC50")
    experimental_description: Optional[str] = dspy.InputField(
        desc="Optional description of experimental details that may be "
             "relevant for predicting the IC50, or None if not available")
    output_unit: str = dspy.InputField(
        desc="The unit of the IC50 value to be predicted, e.g. nM, uM, etc.")
    ic50_true: float = dspy.InputField(
        desc="The experimentally determined IC50 value for the given "
             "drug against the given cell line, in the specified unit."
    )
    colleage_prediction_traces: List[Dict[str, Any]] = dspy.InputField(
        desc="""
        A list of prediction traces from your colleagues, each trace
        containing the following fields:
            - trajectory: A list of dicionaties, each representing a step
                in the prediction process, with keys:
                    - step: The step number (int)
                    - thought: The thought process at this step (str)
                    - tool: The tool used at this step (str)
                    - args: The arguments provided to the tool (str)
                    - observation: The observation or result from the tool (str)
            - ic50_pred: The predicted IC50 value (float)
            - explanation: The final explanation or reasoning for the prediction (str)
            - confidence: The confidence level of the prediction (str)
            - evaluation: A dictionary of evaluation metrics comparing
                the predicted IC50 to the true IC50, e.g. within 2-fold,
                within 3-fold, direction (str: Overestimate/Underestimate/Exact), etc
        """
            )
    existing_drug_instructions: Optional[str] = dspy.InputField(
        desc="""
        Existing instruction set for this drug from the instruction base,
        or "None" / empty if there is no prior record.

        Treat this as a prior hypothesis:
        - Preserve instructions that remain valid under the new evidence.
        - Edit or remove instructions that conflict with consistent new findings.
        - Do NOT blindly append; aim for a compact, non-redundant update.
        """
    )
    existing_cell_line_instructions: Optional[str] = dspy.InputField(
        desc="""
        Existing instruction set for this cell line from the instruction base,
        or "None" / empty if there is no prior record.

        Treat this as a prior hypothesis:
        - Preserve instructions that remain valid under the new evidence.
        - Edit or remove instructions that are contradicted or unhelpful.
        - Do NOT blindly append; aim for a compact, non-redundant update.
        """
    )

    drug_instructions: str = dspy.OutputField(
        desc="""
        The set of instructions for future IC50 predictions
        specific to the drug, as natural language.

        Potential examples of useful instructions: 
        which of the chemical properties appeared to
        be most pertinent to the accuracy of the IC50 prediction, which
        of your colleages's reasoning is the most sound in linking the drug
        properties to the IC50 value, and any pitfalls to avoid in future predictions.
        If you found that the specific drug:cell line interaction to be pertinent
        to provisional IC50 prediction, DO NOT directly include explicit information
        noting about the specific cell line. Instead, please reason about the generalizable
        properties of the cell line (e.g. organism, tissue of origin, disease of origin,
        absence/presence of specific mutations, active pathways, etc.) that
        is potentially responsible for the special interaction, and provide solid
        explained guidance for future predictions and which cell properties to consider.  
        """)
    cell_line_instructions: str = dspy.OutputField(
        desc="""
        The set of instructions for future IC50 predictions
        specific to the cell line, as natural language.
        
        Potential examples of useful instructions: 
        which properties of the cell appeared to
        be most pertinent to the accuracy of the IC50 prediction by your colleagues,
        which of your colleages's reasoning is the most sound in linking the cell
        properties to the IC50 value, and any pitfalls to avoid in future predictions.
        If you found that the specific drug:cell line interaction to be pertinent
        to provisional IC50 prediction, DO NOT directly include explicit information
        noting about the specific drug. Instead, please reason about the generalizable
        properties of the drug (e.g. mechanism of action, target pathways, chemical
        properties, etc.) that is potentially responsible for the special interaction, and provide
        solid explained guidance for future predictions and which drug properties to consider
        when the cell line is being treated.
        """) 
