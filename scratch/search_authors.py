import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.openalex_client import OpenAlexClient
client = OpenAlexClient()

def search_author(name):
    url = f"{client.base_url}/authors"
    params = {
        "filter": f"display_name.search:{name}",
        "sort": "cited_by_count:desc"
    }
    data = client._request_with_retry(url, params=params)
    if data and "results" in data:
        for r in data["results"]:
            print(f"Name: {r['display_name']}, ID: {r['id']}, Cited: {r['cited_by_count']}")

print("Searching Miguel Alcubierre:")
search_author("Miguel Alcubierre")
print("\nSearching Humberto Carrillo Calvet:")
search_author("Humberto Carrillo Calvet")
print("\nSearching Jose Luis Andrade:")
search_author("José Luis Andrade")
