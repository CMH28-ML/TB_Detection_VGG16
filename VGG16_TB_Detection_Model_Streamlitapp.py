import gdown
import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set custom CSS styles for the app
st.markdown(
    """
    <style>
    /* Main content styles */
    .main {
        background-color: #e0f7fa;
        color: #006064;
    }

    /* Title styling */
    h1 {
        color: #004d40;
        font-family: 'Arial Black', sans-serif;
        font-size: 40px;
        margin-bottom: 10px;
    }

    /* Subheader and general text styling */
    .stText {
        font-family: 'Verdana', sans-serif;
        color: #004d40;
        font-size: 18px;
    }

    .stSubheader {
        font-family: 'Arial Black', sans-serif;
        font-size: 28px;
        color: #004d40;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Sidebar styling (dark blue) */
    .sidebar .sidebar-content {
        background-color: #1a237e;
        color: white;
        font-family: 'Verdana';
        font-size: 16px;
        padding: 15px;
    }

    /* Footer styles for the bottom bar */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #004d40;
        color: white;
        text-align: center;
        font-family: 'Arial Black';
        font-size: 24px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.title("PulmoScan AI: Tuberculosis Detection")

# Brief description below title
st.write(
    """
    PulmoScan AI uses advanced deep learning techniques to detect whether a chest X-ray shows signs of tuberculosis.
    """, unsafe_allow_html=True
)

# Upload image section
st.header("Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose a chest X-ray image (JPEG/JPG/PNG)", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Check file size (Streamlit file uploader returns a BytesIO object)
    if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB limit
        st.error("The file is too large. Please upload a file smaller than 10 MB.")
    else:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)
        
        # Load the model (downloaded from Google Drive using gdown)
        @st.cache(allow_output_mutation=True)
        def load_model():
            model_path = 'vgg16_final_model.keras'
            
            # Download the model from Google Drive if not already downloaded
            if not os.path.exists(model_path):
                url = 'https://drive.google.com/uc?id=1m3HKwnDeFi72hqiAy0U2XufiuzAonirY'  # Direct download link
                output = model_path
                gdown.download(url, output, quiet=False)
            
            model = tf.keras.models.load_model(model_path)
            return model
        
        model = load_model()

        # Preprocess image for prediction
        def preprocess_image(image):
            img = Image.open(image).convert('RGB')
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        processed_image = preprocess_image(uploaded_file)
        
        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(processed_image)
            result = "TB Positive" if prediction[0][0] > 0.5 else "Normal"
            st.subheader(f"Prediction: {result}")
            
            # Additional information
            if result == "TB Positive":
                st.error("The image indicates a high probability of tuberculosis. Please consult a doctor.")
            else:
                st.success("The image indicates no signs of tuberculosis.")
            
            # Confidence
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else (1 - prediction[0][0])
            st.write(f"Model confidence: {confidence * 100:.2f}%")
            
            # Show prediction probabilities
            st.write(f"Prediction probabilities: TB Positive: {prediction[0][0] * 100:.2f}%, Normal: {(1 - prediction[0][0]) * 100:.2f}%")

# Sidebar for additional info
st.sidebar.title("About PulmoScan AI")
st.sidebar.info("""
PulmoScan AI is an AI-based tool designed by Chidochashe Monalisa Hodzi to assist in the detection of tuberculosis from chest X-rays. 
This tool is built using a VGG16 model and streamlines the process of diagnosis.
""")

# Footer with additional resources
st.sidebar.title("Additional Resources")
st.sidebar.markdown("""
- [World Health Organization](https://www.who.int/health-topics/tuberculosis)
- [CDC: Tuberculosis Information](https://www.cdc.gov/tb/default.htm)
- [GitHub Repo](https://github.com/CMH28-ML/TB_Detection_VGG16/edit/main/VGG16_TB_Detection_Model_Streamlitapp.py)  
""")

# Footer at the bottom with motivational phrase
st.markdown(
    """
    <div class='footer'>
    We beat TB! PulmoScan AI is here to help.
    </div>
    """, unsafe_allow_html=True
)
