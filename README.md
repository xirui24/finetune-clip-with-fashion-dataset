# Finetune CLIP with Indo fashion dataset
Finetune OpenAI's CLIP model with Indo fashion dataset

## Dataset
We use the Indo fashion dataset available on Kaggle, version15.    
It consists of 106K images and 15 unique cloth categories. The validation and test sets have 500 samples per class.    
An example of the json file:            
```
{'image_url': 'https://m.media-amazon.com/images/I/81LOPbFPiQL._AC_UL320_.jpg',
 'image_path': 'images/val/0.jpeg',
 'brand': 'Generic',
 'product_title': "Women's Khadi Cotton Saree With Blouse Piece (UFO301119SOLD_KHDI_1_Grey)",
 'class_label': 'saree'}
```
To download the dataset: https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset

## Finetuning
We load the pre-trained model from Huggingface, and finetune it with Huggingface Trainer.      
To run the training, load `huggingface_finetune_clip_runner.ipynb` in an environment that already has PyTorch and transformers installed.         
You can modify the training arguments and file paths in the first cells, then run the rest of the notebook.
