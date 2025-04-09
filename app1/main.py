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
    'driver_number', 'lap_number', 'kph', 'top_speed', 'pit_time',
    'driver_stint_no', 'team_stint_no', 'position', 'class_position', 'season_start'
] + [f'circuit_{col}' for col in circuit_columns] + [f'manufacturer_{col}' for col in manufacturer_columns] + ['manufacturer', 'team_no', 'class']

# Load encoders
with open("manufacturer_target_encoding.pkl", "rb") as f:
    manufacturer_target_encoding = pickle.load(f)

with open("team_no_target_encoding.pkl", "rb") as f:
    team_no_target_encoding = pickle.load(f)

# --- Streamlit UI ---
st.title("Lap Time Prediction Input Form")

# Input layout in columns
col1, col2 = st.columns(2)
with col1:
    driver_number = st.number_input("Driver Number", min_value=1, step=1)
    lap_number = st.number_input("Lap Number", min_value=1, step=1)
    kph = st.number_input("Average Speed (kph)", min_value=0.0)
    pit_time = st.number_input("Pit Time (seconds)", min_value=0.0)
    driver_stint_no = st.number_input("Driver Stint Number", min_value=0, step=1)
    position = st.number_input("Position", min_value=1, step=1)

with col2:
    top_speed = st.number_input("Top Speed (kph)", min_value=0.0)
    team_stint_no = st.number_input("Team Stint Number", min_value=0, step=1)
    class_position = st.number_input("Class Position", min_value=1, step=1)
    season_start = st.number_input("Season Start Year", min_value=2012, max_value=2030, step=1)
    car_class = st.selectbox("Car Class", options=class_order)
    manufacturer = st.selectbox("Manufacturer", options=sorted(set(m.replace('_', ' ').title() for m in manufacturer_columns)))
    circuit = st.selectbox("Circuit", options=sorted(set(c.replace('_', ' ').title() for c in circuit_columns)))

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

if st.button("Run Prediction"):
    circuit_clean = re.sub(r'\d+', '', circuit.lower().replace(' ', '_'))
    circuit_clean = re.sub(r'(?i)\b(hours?)\b', '', circuit_clean).strip()
    manufacturer_clean = manufacturer.lower().replace(' ', '_')

    circuit_ohe = {f'circuit_{col}': 0 for col in circuit_columns}
    manufacturer_ohe = {f'manufacturer_{col}': 0 for col in manufacturer_columns}
    circuit_ohe[f'circuit_{circuit_clean}'] = 1
    manufacturer_ohe[f'manufacturer_{manufacturer_clean}'] = 1

    class_encoded = class_order.index(car_class)
    manu_df = pd.DataFrame({'manufacturer': [manufacturer_clean]})
    manufacturer_encoded = manufacturer_target_encoding.transform(manu_df)['manufacturer'].values[0]
    team_df = pd.DataFrame({'team_no': [team_stint_no]})
    team_no_encoded = team_no_target_encoding.transform(team_df)['team_no'].values[0]

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

    encoded_df = pd.DataFrame([final_input])[expected_columns]
    model = load("Lap_Time_Prediction.pkl")
    pred = model.predict(encoded_df)

    # Format the predicted time
    predicted_seconds = pred
    minutes = int(predicted_seconds // 60)
    seconds = int(predicted_seconds % 60)
    milliseconds = int((predicted_seconds - int(predicted_seconds)) * 1000)
    formatted_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    # Show result
    st.subheader("🕒 Predicted Lap Time")
    st.success(f"{formatted_time} (mm:ss:SSS)")



# --- Visualization ---
st.title("Data Visualization from Dataset")

st.subheader("Dataset Insights")
img1, img2 = st.columns(2)
with img1:
    st.image(Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/regression_heatmap.png"), use_container_width=True, caption="Heatmap")
with img2:
    st.image(Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/regression_line.png"), use_container_width=True, caption="Regression Line")

img3, img4 = st.columns(2)
with img3:
    st.image(Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/pitVSlap.png"), use_container_width=True, caption="Pit Time vs Lap Time")
with img4:
    st.image(Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/topSpeedvslaptime.png"), use_container_width=True, caption="Top Speed vs Lap Time")

st.subheader("Min and Max Lap Times")
st.image(Image.open("C:/Users/joeva/Documents/PY_DS/ML_projects/Race/main/graphs/min_max.png"), use_container_width=True, caption="Min and Max Lap times by class and circuit")
