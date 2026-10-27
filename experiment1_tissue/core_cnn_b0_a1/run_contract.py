"""Immutable result identity. Prevent stale fold results from being silently reused."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json

def sha256(path):
    obj=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            obj.update(block)
    return obj.hexdigest()

def contract(inputs, code_paths, config):
    return {'input_sha256':{k:sha256(v) for k,v in inputs.items()},
            'code_sha256':{Path(v).name:sha256(v) for v in code_paths},
            'config':config}

def require_new_or_matching(out, identity, required_results):
    out=Path(out);marker=out/'run_contract.json'
    done=out/'fold_metrics.json'
    if done.exists():
        if not marker.is_file():
            raise RuntimeError(f'Existing metrics have no versioned input+code contract: {done}')
        saved=json.loads(marker.read_text())
        if saved != identity:
            raise RuntimeError(f'Existing fold results have DIFFERENT input/code/config fingerprint: {out}')
        if not all((out/name).is_file() and (out/name).stat().st_size>0 for name in required_results):
            raise RuntimeError(f'Existing fold claims completion but predictions are missing: {out}')
        return True
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError(f'Existing incomplete fold uses another input/code/config fingerprint: {out}')
    out.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(identity,sort_keys=True,indent=2),encoding='utf-8')
    return False
