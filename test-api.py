import requests
import json

# Placeholder for credentials - Use GitHub Secrets in Production
url = ""
api_key = ""

data = {"data": [{"age": 25, "bmi": 28.5, "children": 0}]}
headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

print("Testing endpoint...")
# response = requests.post(url, data=json.dumps(data), headers=headers)
