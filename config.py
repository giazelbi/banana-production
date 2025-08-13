from pathlib import Path

# Path to the data and figure folders inside the package
DATA_PATH = Path(__file__).parent / "data"
FIGURE_PATH = Path(__file__).parent / "figures"

# Ensure they exist (optional)
DATA_PATH.mkdir(exist_ok=True)
FIGURE_PATH.mkdir(exist_ok=True)
