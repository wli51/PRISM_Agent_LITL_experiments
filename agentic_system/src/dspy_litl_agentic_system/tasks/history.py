"""
history.py


"""

from typing import List, Optional

from .task_dispatcher import PrismDispatchQueue

class LITLHistory:
    def __init__(self, dispatchers: List[PrismDispatchQueue]):
        
        if not dispatchers or not isinstance(dispatchers, list):
            raise ValueError(
                f"Expected a non-empty list of PrismDispatchQueue instances, "
                f"got {dispatchers}"
            )
        
        if not all(
            isinstance(dispatcher, PrismDispatchQueue) 
            for dispatcher in dispatchers
        ):
            raise ValueError(
                "All items in dispatchers must be PrismDispatchQueue instances."
            )

        self.dispatchers = dispatchers

    def find_relevant(
        self, 
        drug: Optional[str], 
        cell_line: Optional[str]
    ) -> str:
        
        if not drug and not cell_line:
            raise ValueError(
                "At least one of 'drug' or 'cell_line' must be provided "
                "to find relevant past tasks."
            )

        drug_relevant: List[str] = []
        cell_line_relevant: List[str] = []
        
        for dispatcher in self.dispatchers:
            for key_pair in dispatcher.completed_keys:
                _drug, _cell = key_pair
                if drug and drug == _drug:
                    drug_relevant.append(_cell)
                if cell_line and cell_line == _cell:
                    cell_line_relevant.append(_drug)

        if not drug_relevant and not cell_line_relevant:
            return (
                "No relevant past tasks found involving the specified "
                "drug and cell line. "
                "Proceed with provisional IC50 prediction with general "
                "from exclusively tool calls and/or and general memory search"
            )

        parts: List[str] = ["Relevant past tasks found:"]

        if not drug_relevant or len(drug_relevant) == 0:
            pass
        elif len(drug_relevant) <= 5:
            parts.append(
                f"Tasks predicted with {drug} on cell lines: " \
                    + ', '.join(drug_relevant)
            ) 
        elif len(drug_relevant) > 5:
            parts.append(
                f"Tasks predicted with {drug} on {len(drug_relevant)} cell lines, "
                "including: " + ', '.join(drug_relevant[:5]) + " ..."
            ) 
        else:
            pass

        if not cell_line_relevant or len(cell_line_relevant) == 0:
            pass
        elif len(cell_line_relevant) <= 5:
            parts.append(
                f"Tasks predicted on {cell_line} with drugs: " \
                    + ', '.join(cell_line_relevant)
            ) 
        elif len(cell_line_relevant) > 5:
            parts.append(
                f"Tasks predicted on {cell_line} with {len(cell_line_relevant)} drugs, "
                "including: " + ', '.join(cell_line_relevant[:5]) + " ..."
            )
        else:
            pass

        return '\n'.join(parts) + (
            "\nUse the memory tools with keyworded search to retrieve "
            "details specific to these past tasks if available and needed."
        )
