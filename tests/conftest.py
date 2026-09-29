"""Make the repository root importable (QELM.py, QRC.py, metrics.py, native_reservoirs)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
