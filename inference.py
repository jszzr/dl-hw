import torch
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import json
# from gridattn_image_caption import *
# from Vit_GRU import *

# image_path = './data/flickr8k/images/MEN-Denim-id_00000080-01_7_additional.jpg'  # 替换成你的图像文件路径
# original_image = Image.open(image_path)
# # image_np = original_image.numpy().transpose(1, 2, 0)
# print(original_image)
# plt.imshow(original_image)
# plt.show()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

vocab_path = './data/flickr8k/vocab.json'
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
new_vocab = {v : k for k, v in vocab.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
options = sys.argv[1:]
"""
options[0]: model path
options[1]: image path
"""

def inference(model_path, original_image=None, image_path=None):
    # if options == []:
    #     return
    # model_path = options[0]
    # image_path = options[1]
    if image_path is not None:
        original_image = Image.open(image_path)
    elif original_image is not None:
        pass
    image = preprocess(original_image).unsqueeze(0).reshape(1, 3, 224, 224).to(device)
    checkpoint = torch.load(model_path)
    model = checkpoint['model'].to(device)
    text = model.generate_by_beamsearch(image, 5, 120)
    sentence_list = [new_vocab[i] for i in text[0]]
    sentence_list = sentence_list[1:-1] # 去掉<start>和<end>
    sentence = ' '.join(sentence_list)
    print(sentence)
    return sentence

