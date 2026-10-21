from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"

for _p in (MODELS, DATA, RUNS):
    _p.mkdir(parents=True, exist_ok=True)
