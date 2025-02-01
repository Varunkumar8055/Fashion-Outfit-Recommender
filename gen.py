import os
import json
import faiss
import numpy as np
from PIL import Image
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchvision import transforms, models
import torch
from rembg import remove
from io import BytesIO

# Define paths for input folders and files
image_folder = "images"  # Folder containing clothing item images
embedding_file = "embedding.index"  # Path to save FAISS index
json_file = "captions.json"  # Path to JSON file for metadata

def load_metadata(json_file):
    """
    Load clothing item metadata from a JSON file.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Metadata file '{json_file}' not found.")
    with open(json_file, "r") as f:
        metadata = json.load(f)
    return metadata

def load_deepfashion_embeddings(embedding_file):
    """
    Load precomputed embeddings of clothing items into FAISS.
    """
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    index = faiss.read_index(embedding_file)
    return index

def compute_image_embedding(image, model, preprocess):
    """
    Compute the embedding for the uploaded image using a pre-trained model.
    """
    image_tensor = preprocess(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor).squeeze().numpy()
    return embedding

def recommend_outfits(index, query_embedding, top_k=10):
    """
    Retrieve top-k matching outfits from the FAISS index.
    Returns the indices of the most similar outfits first.
    """
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0], distances[0]

def load_gpt2_model():
    """
    Load the pre-trained GPT-2 model and tokenizer.
    """
    st.info("Loading pre-trained GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def main():
    # Page configuration
    st.set_page_config(page_title="Fashion Outfit Recommender", page_icon="👗", layout="wide")

    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-box {
            border: 2px dashed #FF6347;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .outfit-section {
            background-color: #fff8f0;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .suggestion {
            font-style: italic;
            color: #2F4F4F;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header section
    st.markdown("<div class='main-header'>👗 Fashion Outfit Recommender</div>", unsafe_allow_html=True)

    # Load pre-trained GPT-2 model
    model, tokenizer = load_gpt2_model()

    # Layout for upload section
    st.subheader("Upload a Clothing Image")
    uploaded_image = st.file_uploader("Upload a Clothing Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        # Convert the uploaded image to RGB
        original_image = Image.open(uploaded_image).convert("RGB")
        byte_stream = BytesIO()
        original_image.save(byte_stream, format="PNG")
        
        # Remove background
        bg_removed_image = remove(byte_stream.getvalue())
        bg_removed_image = Image.open(BytesIO(bg_removed_image)).convert("RGB")

        # Display the original and background-removed images side-by-side
        st.image(original_image, caption="Uploaded Image", use_column_width=True)


        # Pre-trained models
        image_model = models.resnet50(pretrained=True)
        image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            index = load_deepfashion_embeddings(embedding_file)
        except FileNotFoundError as e:
            st.error(f"Error: {e}")
            return

        try:
            metadata = load_metadata(json_file)
        except FileNotFoundError as e:
            st.error(f"Error: {e}")
            return

        # Retrieve recommendations
        query_embedding = compute_image_embedding(bg_removed_image, image_model, preprocess)
        recommended_indices, recommended_distances = recommend_outfits(index, query_embedding)

        # Sort recommendations by distance (shortest to longest)
        sorted_recommendations = sorted(
            zip(recommended_indices, recommended_distances), key=lambda x: x[1]
        )

        # Display recommendations in a two-column layout
        st.subheader("Recommended Outfits")
        columns = st.columns(2)  # Create two columns for layout

        for i, (idx, dist) in enumerate(sorted_recommendations):
            with columns[i % 2]:  # Alternate between the two columns
                st.markdown(f"<div class='outfit-section'>", unsafe_allow_html=True)
                st.markdown(f"*Outfit {i + 1}: Distance = {dist:.4f}*")

                # Retrieve description and image
                description = metadata.get(list(metadata.keys())[idx], "No description available.")
                recommended_image_path = os.path.join(image_folder, list(metadata.keys())[idx])
                recommended_image = Image.open(recommended_image_path).convert("RGB")

                # Display image and description
                st.image(recommended_image, caption=f"Outfit {i + 1}", use_column_width=True)
                st.write(f"*Description:* {description}")
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
