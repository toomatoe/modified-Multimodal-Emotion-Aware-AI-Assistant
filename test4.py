import streamlit as st
import cv2
import numpy as np
import keras
from keras.preprocessing import image
import librosa
import tempfile
import os
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai

# --- Initialize ---
recognizer = sr.Recognizer()

# --- Load Models ---
model_best = keras.models.load_model("face_emotion.h5")
audio_model = keras.models.load_model("audio_emotion.keras")

class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
audio_class = ['Fear','Pleasant_surprise', 'Sad', 'angry', 'disgust', 'happy', 'neutral',
               'angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprised', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Gemini Init ---
genai.configure(api_key="AIzaSyBFtqYz55K7TaUHVJ2J05OnjxokTgYRZfk")  # Replace with your actual API key
gemini = genai.GenerativeModel("gemini-2.0-flash")

# --- Functions ---
def predict_face_emotion(frame):
    try:
        print(frame)
        face_img = cv2.resize(frame, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = image.img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        preds = model_best.predict(face_img)
        print("Face Emotion ",preds)
        return class_names[np.argmax(preds)]
    except:
        return "Unknown"

def extract_audio_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = mfccs.T
    if mfccs.shape[0] < 129:
        mfccs = np.pad(mfccs, ((0, 129 - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:129, :]
    return np.expand_dims(mfccs, axis=0)

def predict_audio_emotion(audio):
    try:
        features = extract_audio_features(audio)
        if features is not None:
            preds = audio_model.predict(features)
            return audio_class[np.argmax(preds)]
        return "Unknown"
    except:
        return "Unknown"

def record_until_silence():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Please speak now.")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                with open(temp_wav.name, "wb") as f:
                    f.write(audio_data.get_wav_data())
                return temp_wav.name
            except sr.WaitTimeoutError:
                st.warning("⚠️ Listening timed out.")
                return None

def transcribe_audio_file(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text, np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            print(e)
            return "Sorry, I couldn't understand that.", None

def call_gemini_flash(prompt):
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"

def speak_response(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
    except Exception as ex:
        print(f"Error: {ex}")

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal AI Assistant", layout="centered")
st.title("🧠 Multimodal Emotion-Aware Assistant")

if 'running' not in st.session_state:
    st.session_state.running = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Conversation"):
        st.session_state.running = True
        st.rerun()
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.running = False
        st.rerun()

if st.session_state.running:
    # Capture webcam frame FIRST
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    emotion = "Unknown"
    if ret:
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_reg = frame[y:y + h, x:x + w]
                emotion = predict_face_emotion(face_reg)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Only process first face
            
            st.image(frame, channels="BGR", caption="📸 Captured Frame")
            st.success(f"🧑 Face Emotion: **{emotion}**")
        else:
            st.image(frame, channels="BGR", caption="📸 Captured Frame")
            st.warning("⚠️ No face detected. Please position your face in the camera.")
    else:
        st.error("❌ Couldn't access webcam. Please check camera permissions.")
        st.session_state.running = False
        st.rerun()
    
    # Now record audio
    audio_path = record_until_silence()
    if not audio_path:
        st.warning("No audio recorded.")
        st.session_state.running = False
        st.rerun()

    text, audio_data = transcribe_audio_file(audio_path)
    st.write(f"📝 Transcribed: `{text}`")

    if audio_data is not None:
        audio_emotion = predict_audio_emotion(audio_data)
    else:
        audio_emotion = "Unknown"
    st.success(f"🎙️ Voice Emotion: **{audio_emotion}**")

    # Update conversation history
    st.session_state.conversation_history.append(f"User ({audio_emotion} voice, {emotion} face): {text}")

    # Format conversation history
    formatted_history = ""
    for entry in st.session_state.conversation_history:
        formatted_history += f"{entry}\n"

    # Build Gemini prompt
    conversation_prompt = f"""
You are a warm, conversational AI assistant having a friendly, emotion-aware chat with a human. 
You adapt your tone and content based on the user's facial and voice emotions.

Here is the ongoing conversation so far:
{formatted_history}

Instructions:
- Always stay in-character as a supportive and engaging assistant.
- Detect mood from emotions: 
    - Sad = be comforting
    - Angry = stay calm and de-escalate
    - Happy = match energy and celebrate
    - Neutral = stay friendly
    - Fear/Surprise = be reassuring and patient
- Avoid emojis.
- Keep responses short and natural, unless user asks for long/numbered answers (e.g. "10 points").
- Make your reply feel like the *next natural turn in the conversation*.
- Contantly give comments on user emotions
Respond now as the assistant:
"""

 
    response = call_gemini_flash(conversation_prompt)
    st.write(f"🤖 AI says: {response}")
    speak_response(response)

    st.session_state.conversation_history.append(f"AI: {response}")

    os.remove(audio_path)
    
    # Allow continuous conversation
    st.rerun()
