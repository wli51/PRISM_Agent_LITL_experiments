"""
pubchem_backend.py

Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo

PubChem backends with persistent caching + file-based rate limiting.
Canonical functions are cached by the minimal set of inputs that 
change the HTTP response (e.g., (query), (cid), (cid, threshold)).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import time
import urllib.parse

import httpx

from .utils import FileBasedRateLimiter, tool_cache, get_fetch_limit

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
TIMEOUT = 30.0


class PubChemClient:
    """HTTP client for PubChem API interactions (rate-limited, gentle retry)."""

    def __init__(
        self,
        *,
        max_requests: int = 2,        # PubChem is friendliest <=2 req/s
        time_window: float = 1.0,     # seconds
        rl_name: str = "pubchem"
    ):
        self.client = httpx.Client(
            base_url=PUBCHEM_BASE_URL,
            timeout=TIMEOUT,
            headers={
                "User-Agent": "PubChem-Tools/1.0.0",
                "Accept": "application/json",
            },
        )
        self.rate_limiter = FileBasedRateLimiter(
            max_requests=max_requests, time_window=time_window, name=rl_name
        )

    def _respect_retry_after(self, response: httpx.Response) -> None:
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                delay = float(ra)
                if delay > 0:
                    time.sleep(delay)
            except ValueError:
                time.sleep(1.0)

    def _do(self, method: str, url_or_endpoint: str, **kwargs) -> Dict[str, Any]:
        # Acquire RL token (best-effort)
        try:
            self.rate_limiter.acquire_sync()
        except Exception:
            pass

        # First attempt
        resp = self.client.request(method, url_or_endpoint, **kwargs)
        if resp.status_code in (429, 503):
            self._respect_retry_after(resp)
            try:
                self.rate_limiter.acquire_sync()
            except Exception:
                pass
            resp = self.client.request(method, url_or_endpoint, **kwargs)

        try:
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"API error: {e.response.status_code} - {e.response.text[:200]}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def get(self, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._do("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._do("POST", endpoint, params=params, data=data)

    # For PUG-View (note: different base path)
    def get_view(self, heading: str, cid: Union[int, str]) -> Dict[str, Any]:
        # Use absolute URL; bypass base_url
        url = f"{PUBCHEM_VIEW_BASE_URL}/data/compound/{cid}/JSON"
        # We still want rate limiting + retry on this path:
        return self._do("GET", url, params={"heading": heading})


# Initialize the client and cache “namespace”
pubchem_client = PubChemClient()
cache_name = "pubchem"

# ------------------ Canonical fetchers (cached) ------------------ #

@tool_cache(cache_name)
def _search_pubchem_cids_canonical(query: str) -> Dict[str, Any]:
    """
    Canonical search returning up to a fixed global limit of CIDs.
    Cache key: (query)
    """
    limit = get_fetch_limit()
    endpoint = f"/compound/name/{urllib.parse.quote(query)}/cids/JSON"
    result = pubchem_client.get(endpoint, params={"MaxRecords": limit})
    if "error" in result:
        return {"cids": [], "error": result["error"]}

    cids = result.get("IdentifierList", {}).get("CID", []) or []
    return {"cids": cids[:limit], "error": None}


@tool_cache(cache_name)
def _get_iupac_name_canonical(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Canonical single-name fetch used for enrichment.
    Cache key: (cid)
    """
    endpoint = f"/compound/cid/{cid}/property/IUPACName/JSON"
    result = pubchem_client.get(endpoint)
    if "error" in result:
        return {"name": None, "error": result["error"]}

    props = result.get("PropertyTable", {}).get("Properties", [])
    name = props[0].get("IUPACName") if props else None
    return {"name": name, "error": None}


@tool_cache(cache_name)
def _get_cid_properties_canonical(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Canonical physicochemical properties used elsewhere, cached by (cid).
    """
    props = [
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Complexity",
        "HeavyAtomCount",
        "Charge",
        "CanonicalSMILES",
    ]
    endpoint = f"/compound/cid/{cid}/property/{','.join(props)}/JSON"
    result = pubchem_client.get(endpoint)
    if "error" in result:
        return {"properties": {}, "error": result["error"]}

    table = result.get("PropertyTable", {}).get("Properties", [])
    return {"properties": (table[0] if table else {}), "error": None}


@tool_cache(cache_name)
def _get_assaysummary_canonical(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Canonical assaysummary fetch, cached by (cid).
    """
    endpoint = f"/compound/cid/{cid}/assaysummary/JSON"
    result = pubchem_client.get(endpoint)
    if "error" in result:
        return {"table": {}, "error": result["error"]}
    return {"table": result.get("Table", {}), "error": None}


@tool_cache(cache_name)
def _get_ghs_classification_canonical(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Canonical PUG-View GHS classification fetch, cached by (cid).
    """
    result = pubchem_client.get_view("GHS Classification", cid)
    if "error" in result:
        return {"record": {}, "error": result["error"]}
    return {"record": result.get("Record", {}), "error": None}


@tool_cache(cache_name)
def _get_drug_med_info_canonical(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Canonical PUG-View drug/medication info fetch, cached by (cid).
    """
    result = pubchem_client.get_view("Drug and Medication Information", cid)
    if "error" in result:
        return {"record": {}, "error": result["error"]}
    return {"record": result.get("Record", {}), "error": None}


@tool_cache(cache_name)
def _get_similar_cids_canonical(
    cid: Union[int, str], threshold: int
) -> Dict[str, Any]:
    """
    Canonical similarity search:
    1) fetch SMILES for (cid)
    2) POST similarity search (Threshold, MaxRecords=global_limit*2)
    3) poll if needed
    Cache key: (cid, threshold)  — we return ONLY the CID list; wrappers
    decide how many to display.
    """
    # 1) SMILES
    smi = _get_cid_properties_canonical(cid)
    if smi["error"]:
        return {"similar_cids": [], "error": f"SMILES fetch error: {smi['error']}"}
    smiles = smi["properties"].get("CanonicalSMILES")
    if not smiles:
        return {"similar_cids": [], "error": "No CanonicalSMILES available"}

    # 2) submit
    endpoint = "/compound/similarity/smiles/JSON"
    global_limit = get_fetch_limit()
    params = {"Threshold": int(threshold), "MaxRecords": max(1, global_limit) * 2}
    post_result = pubchem_client.post(endpoint, params=params, data={"smiles": smiles})
    if "error" in post_result:
        return {"similar_cids": [], "error": post_result["error"]}

    # Direct results or wait key
    if "IdentifierList" in post_result:
        found = post_result.get("IdentifierList", {}).get("CID", []) or []
        similar = [c for c in found if str(c) != str(cid)]
        return {"similar_cids": similar, "error": None}

    list_key = post_result.get("Waiting", {}).get("ListKey")
    if not list_key:
        return {"similar_cids": [], "error": "No results and no ListKey returned"}

    # 3) poll
    poll_endpoint = f"/compound/listkey/{list_key}/JSON"
    elapsed = 0
    while elapsed < 30:
        time.sleep(2)
        elapsed += 2
        r = pubchem_client.get(poll_endpoint)
        if "error" in r:
            return {"similar_cids": [], "error": r["error"]}
        if "Waiting" in r or "Fault" in r:
            continue
        found = r.get("IdentifierList", {}).get("CID", []) or []
        similar = [c for c in found if str(c) != str(cid)]
        return {"similar_cids": similar, "error": None}

    return {"similar_cids": [], "error": "Similarity search timed out"}
