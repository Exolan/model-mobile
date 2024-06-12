from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
import json

app = FastAPI()

# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://10.0.2.2:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Загрузка модели и scaler
model = load_model("./my_model.version_0.5")
scaler = joblib.load("scaler_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

def detect_chord_intervals(audio_file):
    y, sr = librosa.load(audio_file)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    change_points = librosa.onset.onset_detect(y=y, sr=sr)
    
    chroma_diff = np.abs(np.diff(chroma.T))
    mean_chroma_diff = np.mean(chroma_diff, axis=1)
    
    threshold = 0.0001
    chord_change_points = [change_points[i] for i in range(len(change_points) - 1) if mean_chroma_diff[i] > threshold]
    
    chord_intervals = [(librosa.frames_to_time(chord_change_points[i]), librosa.frames_to_time(chord_change_points[i + 1]))
                       for i in range(len(chord_change_points) - 1)]

    return chord_intervals

def take_chords(prediction, label_encoder):
    # Создание списка кортежей, содержащих индексы аккордов и их вероятности
    chord_probs = [(i, prob) for i, prob in enumerate(prediction[0])]

    # Сортировка списка по убыванию вероятностей
    sorted_chord_probs = sorted(chord_probs, key=lambda x: x[1], reverse=True)

    # Получение топ 3 самых вероятных аккордов
    top_3_chords = sorted_chord_probs[:3]

    # Создание массива объектов вида {аккорд: вероятность}
    chords_objects = []
    for chord_index, prob in top_3_chords:
        chord = label_encoder.inverse_transform([chord_index])[0]
        chord_obj = {chord: round(prob * 100)}
        chords_objects.append(chord_obj)
    print(chords_objects)
    return chords_objects


def process_interval(audio_file, start_time, end_time, model, scaler, label_encoder):
    try:
        # Извлечение аудиоинтервала
        y, sr = librosa.load(audio_file, mono=True, offset=start_time, duration=end_time-start_time)
        # Извлечение признаков из аудиоинтервала
        rms = np.mean(librosa.feature.rms(y=y))
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
        poly_features = np.mean(librosa.feature.poly_features(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        onset_env = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        # Создание строки с признаками
        result = f"{chroma_stft} {rms} {spec_cent} {spec_bw} {rolloff} {zcr} {tonnetz} {poly_features} {spectral_contrast} {onset_env}"
        result += " ".join([f" {mfcc_val}" for mfcc_val in mfcc])
        
        # Разделение строки на значения и преобразование в массив данных
        values = np.array(result.split(), dtype=float).reshape(1, -1)

        # Масштабирование признаков
        scaled_values = scaler.transform(values)
        # Предсказание аккордов с использованием модели
        prediction = model.predict(scaled_values)

        # Получение вероятных аккордов
        chords = take_chords(prediction, label_encoder)

        return chords

    except Exception as e:
        print(f"An error occurred while processing interval: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_audio_file.wav"  # Получение аудиофайла из запроса
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    chord_intervals = detect_chord_intervals(file_location) # Разбиение файла на отдельные фрагменты
    predictions = []
    for start, end in chord_intervals:
        chords = process_interval(file_location, start, end, model, scaler, label_encoder)
        predictions.append({'start': start, 'end': end, 'chords': chords})

    return json.dumps(predictions)


