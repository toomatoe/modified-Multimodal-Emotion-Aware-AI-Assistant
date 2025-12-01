# 🎭 Multimodal Emotion-Aware AI Assistant

A real-time emotion recognition assistant that uses **facial expressions**, **voice input**, and **LLM-based dialogue generation** to create a personalized, emotion-sensitive user experience. Built with deep learning models and deployed via Streamlit.

---

## 🧠 Features

- 🔊 **Audio Emotion Recognition** using MFCCs + LSTM  
- 📷 **Facial Emotion Detection** using CNN (Xception)  
- 💬 **LLM Integration (Gemini)** for generating emotionally adaptive responses  
- 🎛️ Real-time interface with **Streamlit**, supporting webcam and microphone  
- 🧩 Modular design for scalable and multimodal inputs

---

## 📁 Project Structure


---

## 🔍 How It Works

1. **User provides input** via webcam or microphone.
2. Audio is converted to MFCCs → passed to LSTM model.
3. Facial frames are detected → passed to CNN model.
4. Detected emotion is sent to **Gemini LLM API**.
5. LLM returns a custom-tailored, empathetic reply.
6. Output is displayed on-screen through Streamlit.

---

## 🧪 Models Used

- 🎤 **LSTM-based Audio Classifier**
  - Input: MFCC (40 coefficients/frame)
  - Accuracy: ~95% on TESS Dataset
- 😊 **Xception CNN for Facial Expression Recognition**
  - Fine-tuned on FER-2013 (balanced set)
  - Accuracy: ~88.4%
- 💬 **Gemini LLM** for generating natural language responses

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install -r requirements.txt
```
### ▶️ Run the App

```bash
streamlit run test4.py
```

## 📚 Datasets Used

Audio: TESS Dataset

Images: FER-2013
