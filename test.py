import streamlit as st
import os
import sys

# Use a project-local TensorFlow installation when the default Windows Python
# environment is broken or does not have enough space for the package.
local_packages = os.path.join(os.path.dirname(__file__), ".runtime_packages")
if os.path.isdir(local_packages):
    sys.path.insert(0, local_packages)

from keras.models import load_model
import numpy as np
import cv2 #opencv
from google import genai
from google.genai import types


def get_gemini_client():
    """Create the Gemini client from an environment variable or the project .env file."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            with open(env_path, encoding="utf-8-sig") as env_file:
                for line in env_file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].lstrip()
                    name, separator, value = line.partition("=")
                    if separator and name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                        api_key = value.strip()
                        if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in "\"'":
                            api_key = api_key[1:-1]
                        if api_key:
                            break
        except OSError:
            pass
    if not api_key:
        return None
    # The SDK otherwise waits for its default request timeout, which can leave
    # Streamlit's spinner running for a long time on a stalled connection.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )


def gemini_text(prompt, on_chunk=None):
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY=your_key to the project .env file, "
            "or set GEMINI_API_KEY in your environment."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    answer = ""

    def stream_from(selected_model):
        nonlocal answer
        for chunk in client.models.generate_content_stream(
            model=selected_model,
            contents=prompt,
            config=config,
        ):
            text_chunk = chunk.text or ""
            if text_chunk:
                answer += text_chunk
                if on_chunk:
                    on_chunk(answer)

    try:
        stream_from(model)
    except Exception as exc:
        # Retry transient service overloads or deadlines with Flash-Lite.
        error_text = str(exc).upper()
        is_retryable = any(
            marker in error_text
            for marker in (
                "503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED",
                "TIMED OUT", "TIMEOUT",
            )
        )
        if not is_retryable:
            raise
        if fallback_model == model:
            raise
        # The primary model may already have produced partial text. Clear it
        # before retrying so the fallback response starts cleanly in the UI.
        answer = ""
        if on_chunk:
            on_chunk(answer)
        stream_from(fallback_model)

    if not answer:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return answer.strip()


def pesticides(disease_name, lang, on_chunk=None):
    return gemini_text(
        "You are an agricultural advisor. For the plant disease "
        f"'{disease_name}', give practical, safe management advice in {lang}. "
        "Mention that users should follow local product labels and regulations. "
        "Prefer integrated pest management and only mention pesticides when appropriate.",
        on_chunk=on_chunk,
    )

#function for predicting img 
@st.cache_resource
def get_model():
    return load_model('plant_disease_cnn_model.keras')


def model_predict(image_path):
    model = get_model()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the uploaded image: {image_path}")
    Height, Width, Color = 224, 224, 3 #rgb
    img = cv2.resize(img, (Height, Width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape(1, Height, Width, Color)
    prediction = np.argmax(model.predict(img), axis=-1)[0]

    return prediction 

st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page',['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('Disease.png')

st.image(img)

if(app_mode == 'Home'):
    st.markdown("<h1 style='text-align: center; color : green;'>Plant Disease Prediction System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif(app_mode == 'Disease Recognition'):
    st.header("Plant Disease Prediction System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    lang = st.selectbox("Select language", ['English', 'Telugu', 'Hindi', 'Tamil', 'Kannada', 'Malayalam'])
    save_path = None
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), 'test_image.name')
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

    if st.button("Show Image", disabled=test_image is None):
        st.image(test_image, use_container_width=True)

    if st.button("Predict", disabled=save_path is None):
        st.write("Our Prediction")
        result_index = model_predict(save_path)
        print(result_index)

        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
    
        st.success("Model is predicting that it is a '{}' diesease".format(class_name[result_index]))
        st.session_state['predicted_disease'] = class_name[result_index]

    predicted_disease = st.session_state.get('predicted_disease')
    if st.button("Get Gemini management advice", disabled=not predicted_disease):
        advice_area = st.empty()
        try:
            with st.spinner("Getting advice from Gemini..."):
                advice = pesticides(
                    predicted_disease,
                    lang,
                    on_chunk=lambda partial: advice_area.markdown(partial + " ▌"),
                )
            advice_area.markdown(advice)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
            st.caption(
                "Check that GEMINI_API_KEY is valid, the selected model is enabled "
                "for your API key, and your internet connection is available. "
                "The request is limited to 60 seconds."
            )
    if(st.button("review")):
        review =st.text_input("please give review")
        st.write(f"your response '{review}' is taken. Thank you for giving review about my project") 
    
