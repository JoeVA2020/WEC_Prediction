import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from PIL import Image
from joblib import load
import base64

def set_bg(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg("app1/bg.jpg")

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
    'lap_number', 'kph', 'top_speed', 'pit_time',
    'driver_stint_no', 'team_stint_no', 'position', 'season_start'
] + [f'circuit_{col}' for col in circuit_columns] + [f'manufacturer_{col}' for col in manufacturer_columns] + ['manufacturer', 'team_no', 'class']

# Load encoders
with open("app1/manufacturer_target_encoding.pkl", "rb") as f:
    manufacturer_target_encoding = pickle.load(f)

with open("app1/team_no_target_encoding.pkl", "rb") as f:
    team_no_target_encoding = pickle.load(f)

# --- Streamlit UI ---
st.title("Lap Time Prediction Input Form")

col1, col2 = st.columns(2)
with col1:
    lap_number = st.number_input("Lap Number", min_value=1, step=1)
    st.caption("The current lap number this data refers to.")

    kph = st.number_input("Average Speed (kph)", min_value=0.0)
    st.caption("Average speed of the car during this lap in kilometers per hour.")

    pit_time = st.number_input("Pit Time (seconds)", min_value=0.0)
    st.caption("Total time spent in the pit during this lap, in seconds.")

    driver_stint_no = st.number_input("Driver Stint Number", min_value=0, step=1)
    st.caption("Which stint this is for the current driver.")

    position = st.number_input("Position", min_value=1, step=1)
    st.caption("The car's overall position in the race during this lap.")

with col2:
    top_speed = st.number_input("Top Speed (kph)", min_value=0.0)
    st.caption("Maximum speed reached by the car during this lap in kilometers per hour.")

    team_stint_no = st.number_input("Team Stint Number", min_value=0, step=1)
    st.caption("Team-level stint number.")

    season_start = st.number_input("Season Start Year", min_value=2012, max_value=2030, step=1)
    st.caption("The year when the racing season started.")

    car_class = st.selectbox("Car Class", options=class_order)
    st.caption("The class of the car.")

    manufacturer = st.selectbox("Manufacturer", options=sorted(set(m.replace('_', ' ').title() for m in manufacturer_columns)))
    st.caption("The car's manufacturing brand.")

    circuit = st.selectbox("Circuit", options=sorted(set(c.replace('_', ' ').title() for c in circuit_columns)))
    st.caption("The race track.")

if st.button("Submit"):
    raw_data = {
        'lap_number': [lap_number],
        'kph': [kph],
        'top_speed': [top_speed],
        'pit_time': [pit_time],
        'driver_stint_no': [driver_stint_no],
        'team_stint_no': [team_stint_no],
        'position': [position],
        'season_start': [season_start],
        'class': [car_class],
        'manufacturer': [manufacturer],
        'circuit': [circuit]
    }
    st.subheader("📥 Raw Input Data")
    st.write(pd.DataFrame(raw_data))

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
        'lap_number': lap_number,
        'kph': kph,
        'top_speed': top_speed,
        'pit_time': pit_time,
        'driver_stint_no': driver_stint_no,
        'team_stint_no': team_stint_no,
        'position': position,
        'season_start': season_start,
        'class': class_encoded,
        'manufacturer': manufacturer_encoded,
        'team_no': team_no_encoded,
        **circuit_ohe,
        **manufacturer_ohe
    }

    encoded_df = pd.DataFrame([final_input])[expected_columns]
    model = load("app1/Lap_Time_Prediction.pkl")
    pred = model.predict(encoded_df)

    minutes = int(pred // 60)
    seconds = int(pred % 60)
    milliseconds = int((pred - int(pred)) * 1000)
    formatted_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    st.subheader("🕒 Predicted Lap Time")
    st.success(f"{formatted_time} (mm:ss:SSS)")

with st.sidebar:
    st.title("📊 Data Visualizations")
    st.subheader("Dataset Insights")
    st.image(Image.open("graphs/regression_heatmap.png"), use_container_width=True, caption="Heatmap")
    st.image(Image.open("graphs/regression_line.png"), use_container_width=True, caption="Regression Line")
    st.image(Image.open("graphs/pitVSlap.png"), use_container_width=True, caption="Pit Time vs Lap Time")
    st.image(Image.open("graphs/topSpeedvslaptime.png"), use_container_width=True, caption="Top Speed vs Lap Time")
    st.subheader("Min and Max Lap Times")
    st.image(Image.open("graphs/min_max.png"), use_container_width=True, caption="Min and Max Lap times")
