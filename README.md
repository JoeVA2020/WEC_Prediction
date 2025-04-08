# 📊 FIA WEC Lap Data Projects

## 🎯 Project Objectives

This repository contains **two machine learning projects** based on the [FIA WEC Lap Data (2012–2022)](https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022) dataset. Each project explores a different task:

---

### 📐 1. Lap Time Prediction (Regression)

**Objective:**  
Build regression models to predict lap times using contextual and categorical features such as driver, team, track, car number, etc.

**Preprocessing Steps:**

- Handling Missing Values  
- Encoding Categorical Variables:
  - Label Encoding
  - Target Encoding
  - Ordinal Encoding
  - One-Hot Encoding  
- Datatype Conversion  
- Character Removal  
- List Comprehension  
- Data Visualization  

**Models Used:**

- 📐 Linear Regression  
- 🧭 K-Nearest Neighbors (KNN)  
- 🌲 Decision Tree Regressor  
- 🌳 Random Forest Regressor  
- 📈 Gradient Boosting Regressor  
- ⚡ XGBoost Regressor  

📝 Notebook: `LapTime_Regression.ipynb`

---

### 🏁 2. Racecar Class Prediction (Classification)

**Objective:**  
Predict the class (e.g., LMP1, LMP2, GTE Pro, GTE Am) of a racecar using features from the same dataset.

**Preprocessing Steps:**

- Handling Missing Values  
- Encoding Categorical Variables:
  - Label Encoding
  - Target Encoding
  - One-Hot Encoding  
- Datatype Conversion  
- Character Removal  
- List Comprehension  
- Data Visualization  
- Hyperparameter Tuning  
- Removing Outliers  

**Models Used:**

- 🧭 K-Nearest Neighbors (KNN)  
- 📊 Support Vector Machine (SVM)  
- 🧠 Naive Bayes  
- 🌲 Decision Tree Classifier  
- 🌳 Random Forest Classifier  

📝 Notebook: [`RaceCar_Classification.ipynb`](https://github.com/JoeVA2020/ML_Projects/blob/main/RaceData/RaceCar_Classification.ipynb)

---

## 🗂️ Dataset Information

**Link:** [Kaggle - FIA WEC Lap Data (2012–2022)](https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022)  
**Shape:** `503,679 rows × 48 columns`

This dataset contains:

- Lap time information from FIA WEC (2012–2022)
- Driver and car details
- Car number and class
- Team names
- Track, round, and year metadata

It serves as the foundation for both the regression and classification projects in this folder.
