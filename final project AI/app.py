import streamlit as st
import os
import numpy as np
import librosa
import joblib

# Fungsi untuk ekstraksi fitur
def extract_features(audio_path):
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # Combine features
        features = np.concatenate((
            np.mean(mfcc, axis=1), 
            np.std(mfcc, axis=1),
            [np.mean(pitch), np.std(pitch)],
            [np.mean(spectral_centroid), np.std(spectral_centroid)]
        ))
        
        return features
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Load model dan preprocessing tools
@st.cache_resource
def load_model_and_tools():
    model = joblib.load('gender_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, scaler, encoder

# Fungsi untuk prediksi
def predict_gender(audio_path, model, scaler, encoder):
    feature = extract_features(audio_path)
    if feature is not None:
        feature_scaled = scaler.transform([feature])
        prediction = model.predict(feature_scaled)
        return encoder.inverse_transform(prediction)[0]
    else:
        return None

# Streamlit interface
def main():
    st.title("Gender Detection from Voice")
    st.write("Upload a .wav file to predict the gender (male or female).")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    
    if uploaded_file is not None:
        # Save uploaded file locally
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio("uploaded_audio.wav", format="audio/wav")
        
        # Load model and tools
        model, scaler, encoder = load_model_and_tools()
        
        # Predict gender
        predicted_gender = predict_gender("uploaded_audio.wav", model, scaler, encoder)
        
        if predicted_gender:
            st.success(f"Predicted Gender: {predicted_gender}")
        else:
            st.error("Failed to process the audio file.")

if __name__ == "__main__":
    main()
