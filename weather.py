import streamlit as st
import io
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the model
model_path = ("CnnWeathermodel.h5") 
# encoder_path ="C:/Users/USER/Desktop/MLOPS/encoder.sav"
loaded_model = load_model(model_path,compile=False)

# # Load the encoder
# with open(encoder_path, 'rb') as file:
#     encoder = pickle.load(file)
class_names = ['dew', 'frost', 'glaze', 'lightning', 'rain', 'rainbow', 'sandstorm', 'snow']
encoder = LabelEncoder()
encoder.fit(class_names)
def classifier(uploaded_file):
    # Decode the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Resize and normalize the image
    resize = cv2.resize(image, (64, 64))
    rescaling = resize / 255.0
    rescaling = rescaling.reshape((1, 64, 64, 3))
    
    # Predict the class
    predictor = loaded_model.predict(rescaling, verbose=0)
    predicted_class = np.argmax(predictor, axis=1)
    
    # Convert the predicted class index to the corresponding label using the encoder
    predicted_label = encoder.inverse_transform(predicted_class)
    
    return predicted_label[0]

def main(debug=True):
    # Set page configuration
    st.set_page_config(page_title="Weather Image Detection", page_icon=":seedling:", layout="wide")

    # Custom CSS styles
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4;
            }
            .st-bc {
                color: #333333;
            }
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.title("Weather Image Detection")
    st.markdown("## Upload an image to classify.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display image and classification
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        with col2:
            with st.spinner("Classifying..."):
                # Pass the uploaded file to the classifier
                prediction = classifier(uploaded_file)
            st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()


