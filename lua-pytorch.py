import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torch.utils import serialization
import os
from PIL import Image
from torch.legacy import nn
from vgg_model_places import *

def image_to_places(img_name):
    model = VGGPlaces()
    model.load_weights()

    print(model)

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'http://soundnet.csail.mit.edu/vgg16_places2/categories.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # load the test image
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    #print("input_img", input_img)

    logit = model.forward(input_img)
    #print(logit)
    # forward pass
    h_x = F.softmax(logit).data.squeeze()
    print("h_x", h_x)
    probs, idx = h_x.sort(0, True)

    print('RESULT ON ' + img_name)
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

image_to_places('12.jpg')

image_to_places('12.jpg')