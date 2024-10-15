import gdown
import os
import gdown
import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

# Function to convert image to base64 for embedding in HTML
def get_image_base64(image):
    with open(image, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

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
    }

    /* Subheader and text styling */
    .stText {
        font-family: 'Arial', sans-serif;
        color: #004d40;
    }
    
    /* Prediction result styling */
    .stSubheader {
        font-family: 'Arial Black', sans-serif;
        font-size: 28px;
        color: #004d40;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #004d40;
        color: white;
        font-family: 'Arial';
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

    /* Bubble image styling */
    .bubble-img {
        border-radius: 50%;
        margin-right: 15px;
        float: left;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.title("PulmoScan AI: Tuberculosis Detection")

# Upload the image for the bubble from local machine
uploaded_picture = st.file_uploader("Choose a picture from your machine for display in the bubble", type=["png", "jpg", "jpeg"])

# If a picture is uploaded, convert it to base64 and display in a bubble
if uploaded_picture is not None:
    image_base64 = get_image_base64(uploaded_picture)
    st.write(
        f"""
        <img class='bubble-img' src='data:image/png;base64,{image_base64}' width='100' height='100'/>
        PulmoScan AI uses advanced deep learning techniques to detect whether a chest X-ray shows signs of tuberculosis.
        """, unsafe_allow_html=True
    )
else:
    st.write("PulmoScan AI uses advanced deep learning techniques to detect whether a chest X-ray shows signs of tuberculosis.")

# Upload image section
st.header("Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose a chest X-ray image (JPEG/PNG)", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
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

# Sidebar at the bottom with additional info and motivation
st.markdown(
    """
    <div class='footer'>
    We beat TB! PulmoScan AI is here to help.
    </div>
    """, unsafe_allow_html=True
)

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
