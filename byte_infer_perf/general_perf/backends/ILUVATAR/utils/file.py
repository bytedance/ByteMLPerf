import os
import json

def download_file(src_url, save_path):
    if not os.path.exists(save_path):
        cmd = f"wget {src_url}"
        os.system(cmd)
        
    assert os.path.exists(save_path)
    

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
