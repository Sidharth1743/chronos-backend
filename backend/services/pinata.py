import os
import requests

class PinataUploader:
    def __init__(self, api_key: str, secret_api_key: str):
        """
        Initialize Pinata uploader with API credentials.
        
        Args:
            api_key: Pinata API key
            secret_api_key: Pinata secret API key
        """
        self.api_key = api_key
        self.secret_api_key = secret_api_key
        self.url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
        
    def upload_document(self, file_path: str) -> str:
        """
        Upload a document to IPFS via Pinata.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            IPFS hash/link for the uploaded file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If upload fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as f:
            files = {'file': (os.path.basename(file_path), f)}
            headers = {
                "pinata_api_key": self.api_key,
                "pinata_secret_api_key": self.secret_api_key
            }
            response = requests.post(self.url, files=files, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            ipfs_hash = data.get('IpfsHash')
            if ipfs_hash:
                return f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
            raise ValueError("No IPFS hash in response")
        else:
            raise ValueError(f"Upload failed: {response.text}")
