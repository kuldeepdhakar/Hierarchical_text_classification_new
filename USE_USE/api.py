import requests
import json
title = "Joy Honey & Almonds Advanced Nourishing Body Lotion, For Normal to Dry skin"
text = "The lotion inside the bottle is not the same quality as the make of Joy. It clearly seemed much more less creamy and not smelling good at all."
data = {'title': title, 'text': text}


res = requests.post(url="http://0.0.0.0:7000/api/classify_products", json=data)

print(json.dumps(res.json(), indent=4))
json.dump(res.json(), open("response.json", "w"), indent=4)