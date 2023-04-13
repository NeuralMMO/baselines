# conftest.py
import sys
from pathlib import Path

# Add the 'src' and 'tests' directories to the sys.path
sys.path.extend([
    str(Path(__file__).parent / "tests")
])
