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
    st.caption("Unique identifier for each car.")
    
    kph = st.number_input("Average Speed (kph)", min_value=100.0,step=5)
    st.caption("Average speed of the car during the lap (in kilometers per hour).")

    driver_stint_no = st.number_input("Driver Stint Number", min_value=1, max_value=15, step=1)
    st.caption("The stint (continuous driving period) this driver is currently on.")

    lap_number = st.number_input("Lap Number", min_value=1, max_value=400, step=1)
    st.caption("The lap of the race this data corresponds to.")

    position = st.number_input("Position", min_value=1, step=1,max_value=60 )
    st.caption("The car’s current position in the race.")

    circuit = st.selectbox("Circuit", options=sorted(set(c.replace('_', ' ').title() for c in circuit_columns)))
    st.caption("The race track where the lap was recorded.")
    
    manufacturer = st.selectbox("Manufacturer", options=sorted(set(m.replace('_', ' ').title() for m in manufacturer_columns)))
    st.caption("The car’s manufacturer or constructor.")

with col2:
    driver_number = st.number_input("Driver Number", min_value=1, step=1)
    st.caption("Number assigned to the driver participating in the race.")
    
    top_speed = st.number_input("Top Speed (kph)", min_value=100.0,max_value=400, step=10)
    st.caption("Maximum speed reached by the car during the lap.")

    team_stint_no = st.number_input("Team Stint Number", min_value=1, step=1)
    st.caption("Team’s continuous driving period (can include driver swaps).")

    round_no = st.number_input("Round", min_value=1, max_value=12, step=1)
    st.caption("Round number within the racing season.")
    
    season_start = st.number_input("Season Start Year", min_value=2012, max_value=2030, step=1)
    st.caption("Year when the racing season started.")

    lap_time = st.number_input("Lap Time (seconds)", min_value=60.000)
    st.caption("The total time it took to complete the lap (in seconds).")

    


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

    st.subheader("🏁 Predicted Car Class")
    st.success(f"{pred_class}")

# --- Visualization in Sidebar ---
with st.sidebar:
    st.header("📊 Visual Insights")

    st.image(Image.open("graphs/brfore_Undersample_class.png"), use_container_width=True, caption="Class Distribution before Undersampling")
    st.image(Image.open("graphs/class_distribution.png"), use_container_width=True, caption="Class Distribution After Undersampling")
    st.image(Image.open("graphs/CMD.png"), use_container_width=True, caption="Confusion Matrix of KNN Classifier")
    st.image(Image.open("graphs/topSpeedvslaptime_class.png"), use_container_width=True, caption="Top Speed vs Lap Time")
    st.image(Image.open("graphs/Lap_Time_trend1.png"), use_container_width=True, caption="Lap time trends by class")
    st.image(Image.open("graphs/min_max.png"), use_container_width=True, caption="Min and Max Lap times by class and circuit")
