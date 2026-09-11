"""Import shim: `from scripts.archive.score_phase1 import ...` (archived; used by notebooks/archive/01_phase1_analysis.ipynb) loads scripts/02_score_phase1.py."""
import importlib.util, sys
from pathlib import Path
_p = Path(__file__).resolve().parents[1] / "02_score_phase1.py"
_spec = importlib.util.spec_from_file_location("scripts._score_phase1", _p)
_m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
