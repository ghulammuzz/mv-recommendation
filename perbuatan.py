import requests

url = 'http://127.0.0.1:8000/recommendation/'
data = {"idx": 3} 
response = requests.post(url, json=data)
print(response.json())