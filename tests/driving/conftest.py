import sys
from pathlib import Path

# The driving package only depends on torch, so import it directly from src/
# without installing the full latent_cbf environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "latent_cbf"))
