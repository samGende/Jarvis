import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

if response.status_code == 200:
    with open("downloaded_file.txt", "w") as file:
        file.write(response.text)
    print("File downloaded successfully.")