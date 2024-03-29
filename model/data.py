import json
import pandas as pd
from PIL import Image

def load_data(path):
    """"read json file of dataset"""
    with open(path, 'r') as f:
        data = []
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data


def adjust_dataset_size(path, out_path, n_samples_per_class):
    """sample smaller datasets for quick test, with a balanced distribution for each class label
       path: path of the original dataset
       out_path: path of the new dataset
       n_samples_per_class: number of data to sample for each class label
       """
    data = pd.read_json(path, lines=True)
    small_data = data.groupby('class_label').apply(lambda x: x.sample(n=n_samples_per_class)).reset_index(drop=True)
    small_data.to_json(out_path, orient='records', lines=True)


class DataGenerator:
    def __init__(self, data, data_dir):
        self.data = data
        self.data_dir = data_dir

    def make_batchs(self,batch_size):
        # prepare data for training with LoRA
        N = len(self.data)
        bstart = 0
        while bstart < N :
            b_data = self.data[bstart:bstart+batch_size]
            txt = [d["product_title"] for d in b_data]
            img = [Image.open(self.data_dir+d["image_path"]) for d in b_data]
            yield (txt,img)
            bstart += batch_size

    def get_dataset(self, processor, device):
        # prepare data for trainer
        for d in self.data:
            caption = d["product_title"]
            image_path = d["image_path"]
            image = Image.open(self.data_dir+image_path)
            inputs = processor(text=caption, images=image,return_tensors="pt", max_length=40, padding="max_length", truncation=True).to(device)

            data = {"pixel_values": inputs["pixel_values"].squeeze(),
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()}
        
            yield data
