import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.openalex_data import get_all_works, get_authors

authors = get_authors()
works = get_all_works()
print(f"Authors: {len(authors)}")
print(f"Works: {len(works)}")
