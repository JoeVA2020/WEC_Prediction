import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle

# --- Constants ---
class_order = ['LMP1-H', 'LMP1-L', 'LMP1', 'HYPERCAR', 'LMP2', 'INNOVATIVE CAR', 'LMGTE Pro', 'LMGTE Am', 'CDNT']
circuit_columns = [
    'autodromo_hermanos_rodriguez', 'autodromo_nazionale_di_monza', 'bahrain_international_circuit',
    'circuit_of_the_americas', 'fuji_speedway', 'interlagos', 'le_mans', 'nurburgring', 'sebring',
    'shanghai_international_circuit', 'silverstone', 'spa_francorchamps'
]
manufacturer_columns = [
    'aston_martin', 'audi', 'aurus', 'bmw', 'br01', 'br_engineering', 'chevrolet', 'clm', 'dallara',
    'dome', 'enso', 'ferrari', 'ford', 'gibson', 'ginetta', 'glickenhaus', 'hpd', 'hpd_(honda)',
    'ligier', 'lola', 'morgan', 'nissan', 'norma', 'oak', 'oreca', 'porsche', 'rebellion',
    'riley', 'srt', 'strakka', 'toyota', 'zytek'
]

# Define expected column order (model feature order)
expected_columns = [
    'driver_number', 'lap_number', 'kph', 'top_speed', 'pit_time',
    'driver_stint_no', 'team_stint_no', 'position', 'class_position', 'season_start'
] + \
[c for c in [f'circuit_{col}' for col in circuit_columns]] + \
[m for m in [f'manufacturer_{col}' for col in manufacturer_columns]] + \
['manufacturer', 'team_no', 'class']

# Load encoders
with open("manufacturer_target_encoding.pkl", "rb") as f:
    manufacturer_target_encoding = pickle.load(f)

with open("team_no_target_encoding.pkl", "rb") as f:
    team_no_target_encoding = pickle.load(f)

# --- Streamlit UI ---
st.title("Lap Time Prediction Input Form")

driver_number = st.number_input("Driver Number", min_value=1, step=1)
lap_number = st.number_input("Lap Number", min_value=1, step=1)
kph = st.number_input("Average Speed (kph)", min_value=0.0)
top_speed = st.number_input("Top Speed (kph)", min_value=0.0)
pit_time = st.number_input("Pit Time (seconds)", min_value=0.0)
driver_stint_no = st.number_input("Driver Stint Number", min_value=0, step=1)
team_stint_no = st.number_input("Team Stint Number", min_value=0, step=1)
position = st.number_input("Position", min_value=1, step=1)
class_position = st.number_input("Class Position", min_value=1, step=1)
season_start = st.number_input("Season Start Year", min_value=2012, max_value=2030, step=1)

car_class = st.selectbox("Car Class", options=class_order)
manufacturer = st.selectbox("Manufacturer", options=sorted(set(m.replace('_', ' ').title() for m in manufacturer_columns)))
circuit = st.selectbox("Circuit", options=sorted(set(c.replace('_', ' ').title() for c in circuit_columns)))

# --- Show raw data ---
if st.button("Submit"):
    raw_data = {
        'driver_number': [driver_number],
        'lap_number': [lap_number],
        'kph': [kph],
        'top_speed': [top_speed],
        'pit_time': [pit_time],
        'driver_stint_no': [driver_stint_no],
        'team_stint_no': [team_stint_no],
        'position': [position],
        'class_position': [class_position],
        'season_start': [season_start],
        'class': [car_class],
        'manufacturer': [manufacturer],
        'circuit': [circuit]
    }

    raw_df = pd.DataFrame(raw_data)
    st.subheader("📥 Raw Input Data")
    st.write(raw_df)

# --- Prediction ---
if st.button("Run Prediction"):
    # Preprocess inputs
    circuit_clean = re.sub(r'\d+', '', circuit.lower().replace(' ', '_'))
    circuit_clean = re.sub(r'(?i)\b(hours?)\b', '', circuit_clean).strip()
    manufacturer_clean = manufacturer.lower().replace(' ', '_')

    # One-hot encoding
    circuit_ohe = {f'circuit_{col}': 0 for col in circuit_columns}
    manufacturer_ohe = {f'manufacturer_{col}': 0 for col in manufacturer_columns}
    circuit_ohe[f'circuit_{circuit_clean}'] = 1
    manufacturer_ohe[f'manufacturer_{manufacturer_clean}'] = 1

    # Ordinal encoding
    class_encoded = class_order.index(car_class)

    # Target encoding
    manu_df = pd.DataFrame({'manufacturer': [manufacturer_clean]})
    manufacturer_encoded = manufacturer_target_encoding.transform(manu_df)['manufacturer'].values[0]

    team_df = pd.DataFrame({'team_no': [team_stint_no]})
    team_no_encoded = team_no_target_encoding.transform(team_df)['team_no'].values[0]

    # Combine final input
    final_input = {
        'driver_number': driver_number,
        'lap_number': lap_number,
        'kph': kph,
        'top_speed': top_speed,
        'pit_time': pit_time,
        'driver_stint_no': driver_stint_no,
        'team_stint_no': team_stint_no,
        'position': position,
        'class_position': class_position,
        'season_start': season_start,
        'class': class_encoded,
        'manufacturer': manufacturer_encoded,
        'team_no': team_no_encoded,
        **circuit_ohe,
        **manufacturer_ohe
    }

    encoded_df = pd.DataFrame([final_input])

    # ✅ Reorder columns to match model expectation
    encoded_df = encoded_df[expected_columns]

    # Load model
    with open('Lap_Time_Prediction.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict
    pred = model.predict(encoded_df)

    # Show result
    st.subheader("🕒 Predicted Lap Time")
    st.success(f"{pred[0]:.2f} seconds")

st.title("Data Visualization from Dataset")

from PIL import Image

st.subheader("Heatmap of dataset")
img = Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/regression_heatmap.png")
st.image(img, use_container_width=True)
st.subheader("Regression line plot")
img = Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/regression_line.png")
st.image(img, use_container_width=True)
st.subheader("Pit time VS Lap time plot")
img = Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/pitVSlap.png")
st.image(img, use_container_width=True)
st.subheader("Top speed vs Lap times")
img = Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/topSpeedvslaptime.png")
st.image(img, use_container_width=True)
st.subheader("Min and Max Laptimes")
img = Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/min_max.png")
st.image(img,caption="Min and Max Lap times by class and circuit",use_container_width=True)