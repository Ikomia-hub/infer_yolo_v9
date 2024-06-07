import requests
import os

model_zoo = {
        'yolov9-c': "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt",
        'yolov9-e': "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt",
        'yolov9-s': "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt",
        'yolov9-m': "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt",
        # 'yolov9-t': "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt",
        
    }


def download_model(name, models_folder):
    URL = model_zoo[name]
    print("Downloading model for {}".format(name))
    response = requests.get(URL)
    with open(os.path.join(models_folder, name + ".pt"), "wb") as f:
        f.write(response.content)

