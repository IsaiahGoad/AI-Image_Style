import streamlit as st 
import torch
from torchvision import transforms
from utils import load_image, save_image, get_vgg_model, get_features, gram_matrix
import copy

st.title("🎨 AI-Powered Image Style Transfer")
st.write("Upload a content image and a style image to create your unique artwork!")

# File uploaders
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Display uploaded images
    st.image([content_file, style_file], caption=["Content Image", "Style Image"], width=300)

    # Load images
    content = load_image(content_file)
    style = load_image(style_file)

    # Load model and extract features
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
    vgg = get_vgg_model().to(device).eval()
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target image
    target = content.clone().requires_grad_(True).to(device)

    # Style transfer loop
    optimizer = torch.optim.Adam([target], lr=0.003)
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3}
    content_weight = 1e4
    style_weight = 1e2
    steps = 300

    for step in range(steps):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_loss += style_weights[layer] * torch.mean((target_gram - style_grams[layer]) ** 2) / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            st.write(f"Step {step}/{steps}, Total Loss: {total_loss.item()}")

    # Save and display result
    output_path = "outputs/styled_image.jpg"
    save_image(target, output_path)
    st.image(output_path, caption="Styled Image", width=400)
    st.download_button("Download Styled Image", file_name="styled_image.jpg", data=open(output_path, "rb").read())
