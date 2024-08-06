import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input


def build_autoencoder():
    input_img = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)


    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def encode_images(autoencoder, images):
    encoder = Model(autoencoder.input, autoencoder.get_layer(index=6).output)
    encoded_images = encoder.predict(images)
    encoded_images = encoded_images.reshape((encoded_images.shape[0], -1))  # Flatten the encoded features
    return encoded_images

# Finding similar images
def find_similar_images(encoded_images, image_index, top_n=5):
    distances = np.linalg.norm(encoded_images - encoded_images[image_index], axis=1)
    similar_indices = distances.argsort()[1:top_n+1]
    return similar_indices


def display_recommendations(images, categories, image_index, similar_indices):
    fig, axes = plt.subplots(1, len(similar_indices) + 1, figsize=(20, 20))
    # Original image
    axes[0].imshow(images[image_index])
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Display similar images
    for i, idx in enumerate(similar_indices):
        axes[i+1].imshow(images[idx])
        axes[i+1].set_title(f"Recommendation {i+1} ({categories[idx]})")
        axes[i+1].axis('off')

    st.pyplot(fig)


def display_outfits(tops, bottoms):
    num_outfits = max(len(tops), len(bottoms))
    
    fig, axes = plt.subplots(num_outfits, 2, figsize=(10, num_outfits * 5))
    
    for i in range(num_outfits):

        top_index = i % len(tops)
        bottom_index = i % len(bottoms)

        axes[i, 0].imshow(tops[top_index])
        axes[i, 0].set_title(f"Top {top_index + 1}")
        axes[i, 0].axis('off')

        
        axes[i, 1].imshow(bottoms[bottom_index])
        axes[i, 1].set_title(f"Bottom {bottom_index + 1}")
        axes[i, 1].axis('off')

    st.pyplot(fig)

# Streamlit app
st.set_page_config(
    page_title="Vastra AI",
    page_icon="E:\imp downloads\Codes\VASTRA AI\logo.png", 
    layout="wide"
)


st.title("Vastra AI")
st.markdown("<h2 style='text-align: center; color: #EE82EE;'>Revolutionizing Fashion with Intelligent Style Solutions from Your Own Wardrobe</h2>", unsafe_allow_html=True)
st.write("Upload images and get style recommendations based on the uploaded images.")


uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    images = []
    categories = []  
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).resize((224, 224)).convert('RGB')
        images.append(np.array(img) / 255.0)  

        
        if 'top' in uploaded_file.name.lower():
            categories.append('Top')
        elif 'bottom' in uploaded_file.name.lower():
            categories.append('Bottom')
        else:
            categories.append('Other')

    images = np.array(images)

    
    tops = [images[i] for i in range(len(images)) if categories[i] == 'Top']
    bottoms = [images[i] for i in range(len(images)) if categories[i] == 'Bottom']

    if len(tops) == 0 or len(bottoms) == 0:
        st.write("Please upload both tops and bottoms to generate outfits.")
    else:
        
        autoencoder = build_autoencoder()
        autoencoder.fit(images, images, epochs=50, batch_size=5, shuffle=True, validation_split=0.2)

        
        autoencoder.save('autoencoder_model.keras')

        
        encoded_images = encode_images(autoencoder, images)

        
        st.write("Generated Outfits:")
        display_outfits(tops, bottoms)

        
        for i in range(len(images)):
            st.write(f"Style recommendations for image {i+1} ({categories[i]}):")
            similar_indices = find_similar_images(encoded_images, i)
            display_recommendations(images, categories, i, similar_indices)

        st.write("Style recommendations generated successfully!")