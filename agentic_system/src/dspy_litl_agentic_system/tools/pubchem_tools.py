"""
pubchem_tools.py

Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo

PubChem tools exposed to the agentic system.
All HTTP/caching/ratelimiting is handled in pubchem_backend.py.
These wrappers format human-readable summaries and respect user-facing limits.
"""

from __future__ import annotations
from typing import Dict, Any, List, Union

from .pubchem_backend import (
    _search_pubchem_cids_canonical,
    _get_iupac_name_canonical,
    _get_cid_properties_canonical,
    _get_assaysummary_canonical,
    _get_ghs_classification_canonical,
    _get_drug_med_info_canonical,
    _get_similar_cids_canonical,
)

# --------------------- ID search --------------------- #

def search_pubchem_cid(query: str, limit: int = 5) -> str:
    """
    Search for PubChem CIDs by compound name, CAS number, or formula.
    Returns a human-readable summary of CIDs and key names. 
    If searching by compound name, it is usually best to keep the dashes
    (e.g. "N-Acetylcysteine" instead of "N Acetylcysteine") for better matching.

    Args:
        query (str): The search query (name, CAS, formula).
        limit (int): Maximum number of results to display (default: 5).
    Returns:
        str: Summary of found CIDs or error message.
    """
    data = _search_pubchem_cids_canonical(query)
    if data["error"]:
        return f"Error searching for compound: {data['error']}"
    cids = data["cids"]
    if not cids:
        return f"No compounds found matching '{query}'"

    # If single result, enrich with name
    if len(cids) == 1:
        name = _get_iupac_name_canonical(cids[0])["name"]
        if name:
            return f"Found PubChem CID {cids[0]} ({name}) for '{query}'"
        return f"Found PubChem CID {cids[0]} for '{query}'"

    shown = cids[: max(0, int(limit))]
    return (
        f"Found {len(shown)} compound(s) matching '{query}': CIDs \n - "
        + "\n - ".join(map(str, shown))
    )

# --------------------- Properties --------------------- #

def get_cid_properties(cid: Union[int, str]) -> str:
    """
    Get physicochemical properties for a given PubChem CID. 

    Args:
        cid (int | str): The PubChem Compound ID.
    Returns:
        str: Summary of key properties or error message.
    """
    payload = _get_cid_properties_canonical(cid)
    if payload["error"]:
        return f"Error retrieving properties: {payload['error']}"

    p = payload["properties"]
    if not p:
        return f"No properties found for CID {cid}"

    summary: List[str] = [f"Properties of PubChem CID {cid}:"]

    # Molecular formula + weight
    formula = p.get("MolecularFormula")
    mw = p.get("MolecularWeight")
    if formula and mw:
        try:
            summary.append(f"molecular formula {formula} with MW {float(mw):.2f} g/mol")
        except Exception:
            summary.append(f"molecular formula {formula} with MW {mw} g/mol")

    # Lipophilicity
    xlogp = p.get("XLogP")
    if xlogp is not None:
        try:
            x = float(xlogp)
            lip = "hydrophilic" if x < 0 else ("lipophilic" if x > 3 else "moderate lipophilicity")
            summary.append(f"XLogP {x:.2f} ({lip})")
        except Exception:
            summary.append(f"XLogP {xlogp}")

    # TPSA
    tpsa = p.get("TPSA")
    if tpsa is not None:
        try:
            t = float(tpsa)
            perm = "good" if t < 90 else ("moderate" if t < 140 else "poor")
            summary.append(f"TPSA {t:.1f} Å² ({perm} permeability expected)")
        except Exception:
            summary.append(f"TPSA {tpsa} Å²")

    # H-bonding
    hbd = p.get("HBondDonorCount")
    hba = p.get("HBondAcceptorCount")
    if hbd is not None and hba is not None:
        summary.append(f"{hbd} H-bond donors and {hba} H-bond acceptors")

    # Flexibility
    rtb = p.get("RotatableBondCount")
    if rtb is not None:
        try:
            r = int(rtb)
            flex = "rigid" if r <= 3 else ("flexible" if r >= 7 else "moderate flexibility")
            summary.append(f"{r} rotatable bonds ({flex})")
        except Exception:
            summary.append(f"Rotatable bonds: {rtb}")

    # Complexity
    cplx = p.get("Complexity")
    if cplx is not None:
        try:
            c = float(cplx)
            desc = "simple" if c < 250 else ("complex" if c > 500 else "moderate complexity")
            summary.append(f"molecular complexity {c:.0f} ({desc})")
        except Exception:
            summary.append(f"molecular complexity {cplx}")

    # Charge
    chg = p.get("Charge")
    if chg is not None:
        try:
            ch = int(chg)
            if ch != 0:
                summary.append(f"formal charge {ch:+d} ({'cationic' if ch > 0 else 'anionic'})")
        except Exception:
            pass

    return ". ".join(summary) + "."

# --------------------- Bioassays --------------------- #

def get_bioassay_summary(cid: Union[int, str], max_assays: int = 5) -> str:
    """
    Get bioassay summary for a given PubChem CID.

    Args:
        cid (int | str): The PubChem Compound ID.
        max_assays (int): Maximum number of active assays to list (default: 5).
    Returns:
        str: Summary of bioassay results or error message.
    """
    payload = _get_assaysummary_canonical(cid)
    if payload["error"]:
        return f"Error retrieving bioassay data: {payload['error']}"

    table = payload["table"]
    rows: List[Dict[str, Any]] = table.get("Row", []) or []
    if not rows:
        return f"No bioassay data found for CID {cid}"

    columns: List[str] = table.get("Columns", {}).get("Column", []) or []

    active, inactive, inconclusive = [], [], []
    for row in rows:
        cells = row.get("Cell", []) or []
        assay = {}
        for i, cell in enumerate(cells):
            if i < len(columns):
                assay[columns[i]] = cell
        outcome = str(assay.get("Activity Outcome", ""))
        if "Active" in outcome:
            active.append(assay)
        elif "Inactive" in outcome:
            inactive.append(assay)
        else:
            inconclusive.append(assay)

    parts = [
        f"Bioassay summary for CID {cid}:",
        f"Tested in {len(rows)} assays - {len(active)} active, {len(inactive)} inactive",
    ]

    if active:
        parts.append("\nActive in:")
        for a in active[: max(1, int(max_assays))]:
            aid = a.get("AID", "Unknown")
            name = a.get("Assay Name", "Unknown assay")
            if name and len(name) > 100:
                name = name[:97] + "..."
            parts.append(f"• AID {aid}: {name} \n")

    if len(active) > max_assays:
        parts.append(f"(Showing {max_assays} of {len(active)} active assays)")

    return "\n".join(parts)

# --------------------- Safety (GHS) --------------------- #

def get_safety_summary(cid: Union[int, str]) -> str:
    """
    Get GHS safety classification summary for a given PubChem CID.
    Args:
        cid (int | str): The PubChem Compound ID.
    Returns:
        str: Summary of GHS classification or error message.
    """
    payload = _get_ghs_classification_canonical(cid)
    if payload["error"]:
        return f"No safety data available for CID {cid}"

    record = payload["record"]
    sections = record.get("Section", []) or []
    if not sections:
        return f"No GHS safety classification found for CID {cid}"

    parts: List[str] = [f"Safety information for CID {cid}:"]
    added = False

    def _str_with_markup_list(val) -> List[str]:
        out = []
        if isinstance(val, dict):
            for itm in val.get("StringWithMarkup", []) or []:
                s = itm.get("String")
                if s:
                    out.append(s)
        return out

    for section in sections:
        if section.get("TOCHeading") != "GHS Classification":
            continue
        for sub in section.get("Section", []) or []:
            for info in sub.get("Information", []) or []:
                name = info.get("Name", "")
                value = info.get("Value", {})
                # pictograms
                pictos = _str_with_markup_list(value) if "Pictogram" in str(value) else []
                if pictos:
                    parts.append(f"GHS Pictograms: {', '.join(pictos)}")
                    added = True
                    continue
                # signal word
                if "Signal" in str(value):
                    sw = _str_with_markup_list(value)
                    if sw:
                        parts.append(f"Signal word: {sw[0]}")
                        added = True
                        continue
                # hazards
                if "Hazard Statement" in name:
                    hazards = _str_with_markup_list(value)
                    if hazards:
                        parts.append(f"Hazard statements: {'; '.join(hazards[:3])}")
                        if len(hazards) > 3:
                            parts.append(f"  (and {len(hazards) - 3} more)")
                        added = True

    return "\n".join(parts) if added else f"Limited safety data available for CID {cid}"

# --------------------- Drug & medication info --------------------- #

def get_drug_summary(cid: Union[int, str]) -> str:
    """
    Get drug/medication information summary for a given PubChem CID.
    Args:
        cid (int | str): The PubChem Compound ID.
    Returns:
        str: Summary of drug information or error message.
    """
    payload = _get_drug_med_info_canonical(cid)
    if payload["error"]:
        return f"No drug information available for CID {cid}"

    record = payload["record"]
    sections = record.get("Section", []) or []
    if not sections:
        return f"No drug/medication data found for CID {cid}"

    parts: List[str] = [f"Drug information for CID {cid}:"]
    added = False

    for section in sections:
        for subsection in section.get("Section", []) or []:
            heading = subsection.get("TOCHeading", "") or ""
            infos = subsection.get("Information", []) or []

            if "Therapeutic Use" in heading:
                uses = []
                for info in infos:
                    val = info.get("Value", {})
                    for item in val.get("StringWithMarkup", []) or []:
                        s = item.get("String")
                        if s:
                            uses.append(s)
                if uses:
                    parts.append(f"Therapeutic uses: {', '.join(uses[:3])}")
                    if len(uses) > 3:
                        parts.append(f"  (and {len(uses) - 3} more)")
                    added = True

            elif "Drug Class" in heading:
                classes = []
                for info in infos:
                    val = info.get("Value", {})
                    for item in val.get("StringWithMarkup", []) or []:
                        s = item.get("String")
                        if s:
                            classes.append(s)
                if classes:
                    parts.append(f"Drug classes: {', '.join(classes[:2])}")
                    added = True

            elif "FDA" in heading:
                for info in infos:
                    name = info.get("Name", "")
                    val = info.get("Value", {})
                    s = ""
                    if isinstance(val, dict) and val.get("StringWithMarkup"):
                        s = val["StringWithMarkup"][0].get("String", "")
                    if s and "FDA" in name:
                        parts.append(f"{name}: {s}")
                        added = True

    return "\n".join(parts) if added else f"CID {cid} - no specific drug/medication information available"

# --------------------- Similarity --------------------- #

def find_similar_compounds(
    cid: Union[int, str], threshold: int = 90, max_results: int = 5
) -> str:
    """
    Find compounds similar to the given PubChem CID using a Tanimoto similarity search.
    Args:
        cid (int | str): The PubChem Compound ID.
        threshold (int): Similarity threshold percentage (default: 90).
        max_results (int): Maximum number of similar compounds to display (default: 5).
    Returns:
        str: Summary of similar compounds or error message.
    """
    payload = _get_similar_cids_canonical(cid, int(threshold))
    if payload["error"]:
        return f"Error searching for similar compounds: {payload['error']}"

    similar = payload["similar_cids"]
    if not similar:
        return f"No similar compounds found for CID {cid} at {threshold}% similarity threshold"

    # Slice for display, and enrich with short name/formula if available
    show = similar[: max(1, int(max_results))]
    # Batch names/formulas (best-effort)
    # PubChem supports multiple cids: /property/IUPACName,MolecularFormula/JSON
    try:
        endpoint_props = f"/compound/cid/{','.join(map(str, show))}/property/IUPACName,MolecularFormula/JSON"
        from .pubchem_backend import pubchem_client  # reuse client
        result = pubchem_client.get(endpoint_props)
        lines: List[str] = []
        if "PropertyTable" in result:
            for prop in result["PropertyTable"].get("Properties", []):
                c = prop.get("CID")
                name = prop.get("IUPACName", "No name")
                if name and len(name) > 50:
                    name = name[:47] + "..."
                formula = prop.get("MolecularFormula", "")
                lines.append(f"• CID {c}: {name} ({formula})")
        else:
            lines = [f"• CID {c}" for c in show]
    except Exception:
        lines = [f"• CID {c}" for c in show]

    header = f"Found {len(show)} compounds similar to CID {cid} (≥{threshold}% Tanimoto):"
    return "\n".join([header, *lines])

# --------------------- Exported function list --------------------- #

PUBCHEM_TOOLS = [
    search_pubchem_cid,
    get_cid_properties,
    get_bioassay_summary,
    get_safety_summary,
    get_drug_summary,
    find_similar_compounds,
]
