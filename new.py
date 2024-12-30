import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

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

# Fungsi untuk visualisasi audio
def visualize_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Plot Waveform
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set(title="Waveform of the Uploaded Audio")
        st.pyplot(fig)

        # Plot Mel-Spectrogram
        st.subheader("Mel-Spectrogram")
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='coolwarm')
        ax.set(title="Mel-Spectrogram of the Uploaded Audio")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error visualizing audio file: {e}")

# Load model dan preprocessing tools
@st.cache_resource
def load_model_and_tools():
    model = joblib.load("gender_detection_model.pkl")  # Ganti dengan model Anda
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

# Fungsi untuk prediksi
def predict_genre(audio_path, model, scaler, encoder):
    feature = extract_features(audio_path)
    if feature is not None:
        feature_scaled = scaler.transform([feature])
        prediction = model.predict(feature_scaled)
        probabilities = model.predict_proba(feature_scaled)
        return encoder.inverse_transform(prediction)[0], probabilities
    else:
        return None, None

# Streamlit interface
def main():
    st.title("Genre Detection from Voice")
    st.write("Upload a .wav file to predict the genre and visualize its features.")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    
    if uploaded_file is not None:
        # Save uploaded file locally
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio("uploaded_audio.wav", format="audio/wav")
        
        # Visualize audio
        visualize_audio("uploaded_audio.wav")
        
        # Load model and tools
        model, scaler, encoder = load_model_and_tools()
        
        # Predict genre
        predicted_genre, probabilities = predict_genre("uploaded_audio.wav", model, scaler, encoder)
        
        if predicted_genre:
            st.success(f"Predicted Genre: {predicted_genre}")
            
            # Display probabilities
            st.subheader("Prediction Probabilities")
            prob_df = {genre: prob for genre, prob in zip(encoder.classes_, probabilities[0])}
            st.bar_chart(prob_df)
        else:
            st.error("Failed to process the audio file. Please try again.")

if __name__ == "__main__":
    main()
