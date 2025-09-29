"""
chembl_backends.py

ChEMBL API interaction tools with rate limiting and caching.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo.
"""

from typing import Dict, Any, Optional, List
import time
import httpx

from .utils import FileBasedRateLimiter, tool_cache, get_fetch_limit

# ChEMBL API client configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT = 30.0

class ChEMBLClient:
    """HTTP client for ChEMBL API interactions"""

    def __init__(
        self,
        *,
        max_requests: int = 4,       # up to 4 req/sec is typically safe
        time_window: float = 1.0,    # second(s)
        rl_name: str = "chembl"      # unique name for the RL state file
    ):
        self.client = httpx.Client(
            base_url=CHEMBL_BASE_URL,
            timeout=TIMEOUT,
            headers={
                "User-Agent": "ChEMBL-Tools/1.0.0",
                "Accept": "application/json",
            },
        )
        # --- RATE LIMITER (file-based, cross-process safe)
        self.rate_limiter = FileBasedRateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            name=rl_name,
        )

    def _respect_retry_after(self, response: httpx.Response) -> None:
        """Sleep according to Retry-After header (seconds), if present."""
        ra: Optional[str] = response.headers.get("Retry-After")
        if ra:
            try:
                delay = float(ra)
                if delay > 0:
                    time.sleep(delay)
            except ValueError:
                # If non-numeric (e.g., HTTP-date), just do a small pause
                time.sleep(1.0)

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make GET request to ChEMBL API (rate-limited, with gentle retry)."""

        # ---- RATE LIMIT ACQUIRE (synchronous)
        try:
            self.rate_limiter.acquire_sync()
        except Exception:
            # If RL has a transient issue, fail open but keep going.
            pass

        try:
            response = self.client.get(endpoint, params=params)
            # Soft retry on 429/503 with Retry-After support
            if response.status_code in (429, 503):
                self._respect_retry_after(response)
                # Acquire again before retry to keep RL honest
                try:
                    self.rate_limiter.acquire_sync()
                except Exception:
                    pass
                response = self.client.get(endpoint, params=params)

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"API error: {e.response.status_code} - {e.response.text[:200]}"
            }
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

# Initialize the ChEMBL client
chembl_client = ChEMBLClient()
cache_name = "chembl"

# ---------- chembl id look up ----------

@tool_cache(cache_name)
def _search_chembl_id_canonical(query: str) -> Dict[str, Any]:
    """
    Canonical search that always queries ChEMBL with a fixed global fetch limit.
    Cached by (query) only, independent of user-facing 'limit'.
    Returns a dict: {'compounds': List[str], 'error': Optional[str]}
    """
    global_limit = get_fetch_limit()
    params = {"q": query, "limit": global_limit}
    result = chembl_client.get("/molecule/search.json", params=params)

    if "error" in result:
        return {
            "compounds": [], 
            "error": f"Error searching for compound: {result['error']}"
        }

    molecules = result.get("molecules", [])
    if not molecules:
        return {"compounds": [], "error": None}

    compounds: List[str] = []
    for mol in molecules:  # we already capped at global limit
        chembl_id = mol.get("molecule_chembl_id", "Unknown")
        pref_name = mol.get("pref_name", "No name")
        compounds.append(f"{chembl_id} ({pref_name})")

    return {"compounds": compounds, "error": None}

# ---------- compound properties search ----------

@tool_cache(cache_name)
def _get_compound_properties_canonical(chembl_id: str) -> Dict[str, Any]:
    """
    Canonical property fetch.
    Cached by (chembl_id). Returns raw properties dict for reuse.
    """

    # this query does not need a global limit parameter but we still
    # want to cache the results for re-use

    result = chembl_client.get(f"/molecule/{chembl_id}.json")
    if "error" in result:
        return {
            "properties": {}, 
            "error": f"Error retrieving compound properties: {result['error']}"
        }
    
    molecules = result.get("molecules", [])
    if not molecules:
        return {
            "properties": {}, 
            "error": f"No data found for {chembl_id}"
        }

    props = molecules[0].get("molecule_properties", {})
    if not props:
        return {
            "properties": {}, 
            "error": f"{chembl_id} has no calculated properties available"
        }
    

    return {"properties": props, "error": None, "molecule": molecules[0]}

# ---------- bioactivities search ----------

@tool_cache(cache_name)
def _get_compound_activities_canonical(
    chembl_id: str, activity_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Canonical activity fetch with fixed global limit.
    Cached by (chembl_id, activity_type). Returns raw activities list for reuse.
    """
    global_limit = get_fetch_limit()
    params: Dict[str, Any] = {
        "molecule_chembl_id": chembl_id, "limit": global_limit}
    if activity_type:
        params["standard_type"] = activity_type

    result = chembl_client.get("/activity.json", params=params)
    if "error" in result:
        return {
            "activities": [], 
            "error": f"Error retrieving bioactivities: {result['error']}"
        }

    return {"activities": result.get("activities", []), "error": None}

# ---------- drug approval/info search ----------

@tool_cache(cache_name)
def _get_drug_info_canonical(chembl_id: str) -> Dict[str, Any]:
    """
    Canonical general drug info fetch.
    Cached by (chembl_id). Returns raw drug info dict for reuse.
    """

    global_limit = get_fetch_limit()
    result = chembl_client.get(
        "/drug.json", 
        params={"molecule_chembl_id": chembl_id, "limit": global_limit}
    )

    if "error" in result:
        return {
            "info": None, 
            "error": f"Error retrieving drug info: {result['error']}"
        }
    
    return {
        "info": result,
        "error": None
    }

# ---------- drug moa search ----------

def _get_drug_moa_canonical(chembl_id: str) -> Dict[str, Any]:
    """
    Canonical mechanism of action fetch.
    Cached by (chembl_id). Returns raw MoA list for reuse.
    """

    global_limit = get_fetch_limit()
    result = chembl_client.get(
        "/mechanism.json", 
        params={"molecule_chembl_id": chembl_id, "limit": global_limit}
    )
    if "error" in result:
        return {
            "moa": [], 
            "error": f"Error retrieving mechanism of action: {result['error']}"
        }
    
    return {
        "moa": result,
        "error": None
    }

# ---------- drug indications search ----------

def _get_drug_indications_canonical(chembl_id: str) -> Dict[str, Any]:
    """
    Canonical drug indications fetch.
    Cached by (chembl_id). Returns raw indications list for reuse.
    """

    global_limit = get_fetch_limit()
    result = chembl_client.get(
        "/drug_indication.json", 
        params={"molecule_chembl_id": chembl_id, "limit": global_limit}
    )
    if "error" in result:
        return {
            "indications": [], 
            "error": f"Error retrieving indications: {result['error']}"
        }
    
    return {
        "indications": result,
        "error": None
    }

# ---------- target id search ----------
def _search_target_id_canonical(query: str) -> Dict[str, Any]:
    """
    """
    global_limit = get_fetch_limit()
    params = {"q": query, "limit": global_limit}
    result = chembl_client.get("/target/search.json", params=params)

    if "error" in result:
        return {
            "targets": [], 
            "error": f"Error searching for target: {result['error']}"
        }
    
    return {
        "targets": result, 
        "error": None
    }

def _get_target_activities_summary_canonical(
    target_chembl_id: str,
    activity_type: str | None = "IC50",
):
    """
    Canonical target activities summary fetch with fixed global limit.
    Cached by (target_chembl_id, activity_type). 
    Returns raw activities list for reuse.
    """
    global_limit = get_fetch_limit()
    params = {
        "target_chembl_id": target_chembl_id,
        "limit": global_limit,
    }
    if activity_type:
        params["standard_type"] = activity_type
    result = chembl_client.get("/activity.json", params=params)
    
    if "error" in result:
        return {
            "activities": [], 
            "error": f"Error retrieving bioactivities: {result['error']}"
        }

    return {
        "activities": result, 
        "error": None
    }
