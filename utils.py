import torch
from torchvision import models, transforms
from PIL import Image
import copy

# Preprocessing function for images
def load_image(image_file, max_size=400):
    image = Image.open(image_file).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return transform(image).unsqueeze(0)

# Postprocessing function to display images
def save_image(tensor, path):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

# Load VGG19 model
def get_vgg_model():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

# Extract content and style features
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Compute Gram Matrix for style features
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram
