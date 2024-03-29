# Finetune CLIP with Indo fashion dataset
Finetune OpenAI's CLIP model with Indo fashion dataset    
We offer an option to use LoRA during training

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
We'll use the images and `procut_title` to train.        

To download the dataset: https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset

## Finetuning
We load the pre-trained model from Huggingface, and finetune it with Huggingface Trainer.      
If you want to train with LoRA, set `train_with_lora` to True.    
LoRA: https://arxiv.org/abs/2103.00020

To run the training, load `huggingface_finetune_clip_runner.ipynb` in an environment that already has PyTorch and transformers installed.         
You can modify the training hyperparameters and file paths in the first cell, then run the rest of the notebook.

## Evaluation
The evaluation task is to predict the cloth categories (`class_label`).       
The prediction is to mesure the similarity of the images and texts ("a photo of `{label}`") features calculated by the model, and return the k labels with highest similarities.            
We'll firstly evalute before training, the result can be considered as a baseline, then do another evaluation after training.     
