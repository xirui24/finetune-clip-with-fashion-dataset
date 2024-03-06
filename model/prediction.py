import torch

def get_features(texts, images, model, processor, device):
    """Get text and image features"""

    with torch.no_grad():

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)

    text_features = outputs.text_embeds
    image_features = outputs.image_embeds

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features, image_features


def image_features(images, model, processor, device):
    """Get only image features"""

    with torch.no_grad():

        inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


def make_single_prediction(text_features, image_features, k, classes):
    """Make top-k predictions for a single image
       text_features, image_features: text and image features calculated by get_features() or image_features()
       k: integer
       classes: all class labels of the dataet, list of strings"""

    similarity = text_features @ image_features.T
    topk_sim, topk_indices = torch.topk(similarity.flatten(), k)

    print(f"Top {k} predictions: ")
    for sim, indice in zip(topk_sim, topk_indices):
        print(f"    {classes[indice]}: {sim*100:.2f}%")


def make_full_prediction(text_features, image_features, gold_labels, k):
    """Calculate top-k accuracy for all images
       text_features, image_features: text and image features calculated by get_features() or image_features()
       correct_labels: correct labels of the images, list of integers
       k: integer"""
    
    predictions = []
    
    for i in range(len(text_features)):
        similarity =  image_features[i] @ text_features.T # calculate similarity between each image with all labels
        indices = torch.topk(similarity.flatten(), k).indices
        predictions.append((gold_labels[i],indices))
    
    correct = 0
    for i, indices in predictions:
        if i in indices:
            correct += 1
    
    accuracy = correct/len(text_features)

    return format(accuracy,".4f")