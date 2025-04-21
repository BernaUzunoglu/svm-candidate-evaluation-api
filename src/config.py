import os
from pathlib import Path

class Config:
    # PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app/"))
    # Ortam değişkeninden al, yoksa bu dosyanın 2 üst klasörünü baz al
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))