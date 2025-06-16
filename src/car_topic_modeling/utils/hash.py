from dataclasses import asdict
import hashlib
import json
from typing import Any, Dict
import logging

log = logging.getLogger(__name__)


def build_hash_with_params(params: Any) -> str:
    params_dict: Dict[str, Any]
    if not isinstance(params, dict):
        params_dict = asdict(params)
    else:
        params_dict = params
    log.info(f"Building hash name with params object: {params_dict}")
    blob = json.dumps(params_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:8]
