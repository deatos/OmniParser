import requests
import base64

# Define the URL of the API server
url = "http://127.0.0.1:8005/infer"

# Read the image file and encode it in base64
with open("imgs/google_page.png", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

# Define the payload for the POST request
payload = {
    "prompt": "",
    "image_base64": image_base64
}

# Send the POST request to the API server
response = requests.post(url, json=payload)

# Print the response from the server
print(response.json())