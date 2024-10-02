import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Title and description
st.title("PulmoScan AI: Tuberculosis Detection")
st.write("PulmoScan AI uses advanced deep learning techniques to detect whether a chest X-ray shows signs of tuberculosis.")

# Sidebar for additional info
st.sidebar.title("About PulmoScan AI")
st.sidebar.info("""
PulmoScan AI is an AI-based tool designed to assist in the detection of tuberculosis from chest X-rays. 
This tool is built using a VGG16 model and streamlines the process of diagnosis.
""")

# Upload image section
st.header("Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose a chest X-ray image (JPEG/PNG)", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)
    
    # Load the model (from Google Drive or any hosted link)
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('https://drive.google.com/file/d/1m3HKwnDeFi72hqiAy0U2XufiuzAonirY/view?usp=drive_link')  # Actual link
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
        
        # Confidence and probability
        probability = prediction[0][0]
        confidence = probability if probability > 0.5 else (1 - probability)
        st.write(f"Model confidence: {confidence * 100:.2f}%")
        st.write(f"Prediction Probability: **{probability:.2f}** (TB Positive likelihood)")

# Footer with additional resources
st.sidebar.title("Additional Resources")
st.sidebar.markdown("""
- [World Health Organization](https://www.who.int/health-topics/tuberculosis)
- [CDC: Tuberculosis Information](https://www.cdc.gov/tb/default.htm)
- [GitHub Repo](https://github.com/CMH28-ML/TB_Detection_VGG16/edit/main/VGG16_TB_Detection_Model_Streamlitapp.py)  
""")
