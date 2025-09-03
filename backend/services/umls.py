import requests
import os
from dotenv import load_dotenv

class UMLSService:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("UMLS_API_KEY")
        if not self.api_key:
            raise ValueError("UMLS_API_KEY not found in environment variables")
        self.base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"

    def get_umls_concept(self, term):
        """Get UMLS CUI and semantic type for a given term."""
        if not term:
            return None

        params = {
            "apiKey": self.api_key,
            "string": term,
            "searchType": "exact",
            "returnIdType": "concept",
            "sabs": "SNOMEDCT_US,MESH",
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            results = data.get("result", {}).get("results", [])
            
            if results and results[0].get("ui") != "NONE":
                return {
                    'cui': results[0]["ui"],
                    'semantic_type': results[0].get("semanticType", "Unknown"),
                    'preferred_name': results[0].get("name", term)
                }
            return None
            
        except Exception as e:
            print(f"UMLS API error for term '{term}': {str(e)}")
            return None
