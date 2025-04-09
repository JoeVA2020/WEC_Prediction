import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from PIL import Image
from joblib import load

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

expected_columns = [
    'car_number', 'driver_number', 'lap_number', 'kph', 'top_speed', 'round',
    'lap_time_s', 'driver_stint_no', 'team_stint_no', 'position', 'season_start'
] + [f'circuit_{col}' for col in circuit_columns] + [f'manufacturer_{col}' for col in manufacturer_columns] + ['manufacturer_encoded', 'team_encoded']

# Load encoders
with open("app2/team_encoder.pkl", "rb") as f:
    team_enc = pickle.load(f)

with open("app2/manufacturer_encoder.pkl", "rb") as f:
    manufacturer_enc = pickle.load(f)

# --- Streamlit UI ---
st.title("🏎️ Car Class Classification")

# Input layout in columns
col1, col2 = st.columns(2)
with col1:
    car_number = st.number_input("Car Number", min_value=1, step=1)
    driver_number = st.number_input("Driver Number", min_value=1, step=1)
    lap_number = st.number_input("Lap Number", min_value=1, max_value=400, step=1)
    kph = st.number_input("Average Speed (kph)", min_value=100.0)
    driver_stint_no = st.number_input("Driver Stint Number", min_value=1, max_value=10, step=1)
    position = st.number_input("Position", min_value=1, step=1)
    circuit = st.selectbox("Circuit", options=sorted(set(c.replace('_', ' ').title() for c in circuit_columns)))

with col2:
    top_speed = st.number_input("Top Speed (kph)", min_value=100.0)
    team_stint_no = st.number_input("Team Stint Number", min_value=1, step=1)
    season_start = st.number_input("Season Start Year", min_value=2012, max_value=2030, step=1)
    lap_time = st.number_input("Lap Time (seconds)", min_value=60.000)
    round_no = st.number_input("Round", min_value=1, max_value=12, step=1)
    manufacturer = st.selectbox("Manufacturer", options=sorted(set(m.replace('_', ' ').title() for m in manufacturer_columns)))

# --- Prediction ---
if st.button("Run Classification"):
    # Normalize inputs
    circuit_clean = re.sub(r'\d+', '', circuit.lower().replace(' ', '_'))
    circuit_clean = re.sub(r'(?i)\b(hours?)\b', '', circuit_clean).strip()
    manufacturer_clean = manufacturer.lower().replace(' ', '_')

    # One-hot encode circuit and manufacturer
    circuit_ohe = {f'circuit_{col}': 0 for col in circuit_columns}
    manufacturer_ohe = {f'manufacturer_{col}': 0 for col in manufacturer_columns}
    circuit_ohe[f'circuit_{circuit_clean}'] = 1
    manufacturer_ohe[f'manufacturer_{manufacturer_clean}'] = 1

    # Target encoding
    team_df = pd.DataFrame({'team': [team_stint_no]})
    manu_df = pd.DataFrame({'manufacturer': [manufacturer_clean]})
    team_encoded = team_enc.transform(team_df)['team'].values[0]
    manufacturer_encoded = manufacturer_enc.transform(manu_df)['manufacturer'].values[0]

    # Construct input
    final_input = {
        'car_number': car_number,
        'driver_number': driver_number,
        'lap_number': lap_number,
        'kph': kph,
        'top_speed': top_speed,
        'round': round_no,
        'lap_time_s': lap_time,
        'driver_stint_no': driver_stint_no,
        'team_stint_no': team_stint_no,
        'position': position,
        'season_start': season_start,
        'manufacturer_encoded': manufacturer_encoded,
        'team_encoded': team_encoded,
        **circuit_ohe,
        **manufacturer_ohe
    }
    

    scaler = load("app2/minmax_scaler.pkl")
    encoded_df = pd.DataFrame([final_input])[expected_columns]
    scaled_df = pd.DataFrame(scaler.transform(encoded_df), columns=expected_columns)


    # Load model and predict
    model = load("app2/RandomForestClassifier.pkl")
    pred_class = model.predict(scaled_df)
    print(pred_class)
    # Show result
    st.subheader("🏁 Predicted Car Class")
    st.success(f"{pred_class}")
