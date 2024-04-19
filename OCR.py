import os
from dotenv import load_dotenv
import requests
import time

load_dotenv()

HF_TOKEN=os.getenv('HF_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-large-handwritten"
headers = {"Authorization": "Bearer "+HF_TOKEN}

def query(filename):
    print("going to query OCR")
    time.sleep(4)
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()