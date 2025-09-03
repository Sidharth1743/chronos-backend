import requests
import os
from dotenv import load_dotenv

load_dotenv()
umls_api_key = os.getenv("UMLS_API_KEY")

def get_umls_cui(term):
    base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "apiKey": umls_api_key,
        "string": term,
        "searchType": "exact",
        "sabs": "SNOMEDCT_US,MESH",  # Include multiple vocabularies
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("result", {}).get("results", [])
            print("API Response:", data)  # Debug: Print full response
            if results and results[0].get("ui") != "NONE":
                cui = results[0]["ui"]
                # Use get() to handle missing semanticType
                semantic_type = results[0].get("semanticType", "Unknown")
                return cui, semantic_type
            return None, None
        else:
            print(f"API Error: Status {response.status_code}, {response.text}")
            return None, None
    except Exception as e:
        print(f"UMLS API error for {term}: {e}")
        return None, None

if __name__ == "__main__":
    cui, semantic_type = get_umls_cui("spina bifida occulta")
    print(f"CUI: {cui}, Semantic Type: {semantic_type}")