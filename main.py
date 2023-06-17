from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import librosa
import numpy as np

app = Flask(__name__)
model = load_model('Speech-Emotion-Recognition-Model.h5')
labels = ['disgust', 'happy', 'sad', 'neutral', 'fear', 'angry']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        data, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = model.predict(mfcc)
        predicted_label = labels[np.argmax(prediction)]
        return render_template('index.html', emotion_result=f"<h1>Predicted Emotion: {predicted_label}</h1>")
    except:
        return jsonify({'error': 'Error occurred during prediction'})


if __name__ == '__main__':
    app.run(debug=True)