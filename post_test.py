import requests

BASE = 'http://185.243.54.180:8080/'

with open('shield_raw_base64.txt', 'r') as file:
    img64 = file.read()

#print(img64)

response = requests.post(BASE + "training/2", {"training_id": 2, "image": img64, "caliber": 9,
                                               "magazine_capacity": 15, "distance_to_target": 15})
print(response.json())
