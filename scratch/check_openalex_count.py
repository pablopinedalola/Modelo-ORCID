import requests
import json

def check_count():
    url = "https://api.openalex.org/authors"
    params = {
        "filter": "last_known_institutions.country_code:MX",
        "per_page": 1
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Total authors in Mexico: {data['meta']['count']}")
        else:
            print(f"Error: {resp.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    check_count()
