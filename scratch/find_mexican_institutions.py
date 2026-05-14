import requests

INSTITUTIONS = [
    "Universidad Nacional Autónoma de México",
    "Instituto Politécnico Nacional",
    "Universidad Autónoma Metropolitana",
    "Centro de Investigación y de Estudios Avanzados del Instituto Politécnico Nacional",
    "Tecnológico de Monterrey"
]

def find_institutions():
    for name in INSTITUTIONS:
        url = "https://api.openalex.org/institutions"
        params = {"search": name, "per_page": 1}
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                inst = results[0]
                print(f"{name}: {inst['id']} (ROR: {inst.get('ror')})")
            else:
                print(f"{name}: Not found")

if __name__ == "__main__":
    find_institutions()
