import base64
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# ✅ Page Config
st.set_page_config(page_title="Teeth Classification App", layout="centered")

# ✅ Title
st.title("🦷 Teeth Classification - AI Model")
st.write("Upload a dental image and the AI will predict the disease of the tooth.")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded_string.decode()}");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_local(r'D:\Cellula Technologies\Project 1\Python\teeth_classification_app\background3.jpeg')



# ✅ Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("teeth_resnet50_finetuned.h5")
    return model

model = load_model()
class_names = ['Caries (Dental Caries)', 'Calculus (Dental Calculus / Tartar)', 'Gingivitis', 'Mucosal Cyst', 'Oral Cancer', 'Oral Lichen Planus', 'Other']

# ✅ Upload Image
uploaded_file = st.file_uploader('Upload an Image...', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # ✅ Preprocess - هنا هستخدم preprocess_input
    img = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)   # ✅ هنا المهمة
    

    # ✅ Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    
    
    st.success(f"**Prediction:** {class_names[class_idx]}")
    st.info(f"**Confidence:** {confidence:.2f}%")



    
