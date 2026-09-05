"""Single place to choose the model and the thinking switch.

MODEL_ID comes from the MATS_MODEL_ID environment variable (also read from a `.env` file in
the repo root if present), defaulting to the laptop model. The pod sets
MATS_MODEL_ID=Qwen/Qwen3.5-9B in its venv activation script, so the same committed file
works everywhere and config no longer diverges between machines.
"""
import os
from pathlib import Path

_LAPTOP_DEFAULT = "Qwen/Qwen3-1.7B"   # dense, ~3.4GB bf16, fine on a 16GB Apple Silicon laptop
# GPU options (both Qwen3_5ForConditionalGeneration hybrids; need fla + causal_conv1d, see docs/pod_setup.md):
#   Qwen/Qwen3.5-9B    primary: 32 layers, hidden 4096, ~18GB bf16
#   Qwen/Qwen3.6-27B   replication on an 80GB card: 64 layers, hidden 5120, ~54GB bf16


def _load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Minimal KEY=VALUE loader; existing environment variables win over the file."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()
MODEL_ID = os.environ.get("MATS_MODEL_ID", _LAPTOP_DEFAULT)

# Qwen3 / Qwen3.5 / Qwen3.6 chat templates have a thinking mode. With enable_thinking=False the
# template closes an empty <think></think> block in the *prompt*, so the reply starts directly
# with the answer. chat() and steer_generate() both read this flag by default.
ENABLE_THINKING = os.environ.get("MATS_ENABLE_THINKING", "0") == "1"
