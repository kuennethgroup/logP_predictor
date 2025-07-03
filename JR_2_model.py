import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import pickle
import matplotlib.pyplot as plt

# 1. Parquet-Datei laden
parquet_path = "./data_0604_100_10.parquet" 
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"Parquet-Datei fehlt oder Pfad falsch: {parquet_path}")

df = pd.read_parquet(parquet_path)
print("Spalten: ", df.columns.tolist())

# 2. Nur relevante Spalten behalten
df = df[['copolymers_logP', 'copolymers_fingerprints', 'n', 'm', 'x']].copy()

# 3. Fingerprints in einzelne Spalten aufteilen
print("Beispiel-Fingerprint: ", df['copolymers_fingerprints'].iloc[0])
print("Länge: ", len(df['copolymers_fingerprints'].iloc[0]))

fp_df = pd.DataFrame(df['copolymers_fingerprints'].tolist())

# 4. Neues DataFrame für ML
df_ml = pd.concat([df[['copolymers_logP', 'n', 'm', 'x']], fp_df], axis=1)
print(df_ml.head())

# 5. NaN entfernen
df_ml = df_ml.dropna()
print("Shape nach NaN-Drop: ", df_ml.shape)
print("Gesamtanzahl Daten für ML: ", len(df_ml))

from sklearn.preprocessing import StandardScaler

# 5a. Zielvariable skalieren
scaler_logP = StandardScaler()
df_ml['copolymers_logP_scaled'] = scaler_logP.fit_transform(df_ml[['copolymers_logP']])

# 5b. n, m, x skalieren
scaler_nmx = StandardScaler()
df_ml[['n_scaled', 'm_scaled', 'x_scaled']] = scaler_nmx.fit_transform(df_ml[['n', 'm', 'x']])

# 5c. Unskalierte Spalten entfernen
df_ml = df_ml.drop(columns=['n', 'm', 'x'])

# 6. Train-Test-Split
train_data, test_data = train_test_split(df_ml, test_size=0.2, random_state=42)
print("Trainingsdaten: ", train_data.shape)
print("Testdaten: ", test_data.shape)

# Entferne das unskalierte Target aus den Trainings- und Testdaten für AutoGluon:
train_data_ag = train_data.drop(columns=['copolymers_logP'])   
test_data_ag = test_data.drop(columns=['copolymers_logP'])

# 7. Features und Label trennen
# Skalierte n, m, x
scaled_cols = ['n_scaled', 'm_scaled', 'x_scaled']
# Fingerprint-Spalten (alle Integer-Spalten in df_ml)
fp_cols = [col for col in df_ml.columns if isinstance(col, int)]
feature_cols = scaled_cols + fp_cols

X_train = train_data_ag[feature_cols]   # nur skalierte Features
y_train = train_data_ag['copolymers_logP_scaled']
X_test = test_data_ag[feature_cols]
y_test = test_data_ag['copolymers_logP_scaled']                    

# 8. Speichern
save_dir = "./JR_important/"

os.makedirs(save_dir, exist_ok=True)

with open(save_dir + 'X_train_unit.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open(save_dir + 'X_test_unit.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(save_dir + 'y_train_unit.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open(save_dir + 'y_test_unit.pkl', 'wb') as f:
    pickle.dump(y_test, f)                                         

# 9. Hyperparameter für das NN-Modell (Aktivierungsfunktion ändern)
from autogluon.common import space as ag  

hyperparameters = {
    'NN_TORCH': {
        'activation': ag.Categorical('relu', 'gelu', 'softrelu')   
    }
}

# 10. Modelltraining mit Autogluon
predictor = TabularPredictor(
    label='copolymers_logP_scaled',
    problem_type='regression',
    path='./JR_important/all_predictions_0601'
)
predictor.fit(
    train_data=train_data_ag,
    presets='best_quality',      # maximale Genauigkeit, viele Modelle
    time_limit=14400,             # ! Zeit anpassen
    dynamic_stacking=True,
    hyperparameters=hyperparameters,
    num_bag_folds=8              # für beste Qualität
)

# 11. Vorhersage und Auswertung
y_pred_scaled = predictor.predict(X_test)                              
y_pred = scaler_logP.inverse_transform(y_pred_scaled.values.reshape(-1, 1)).flatten() 
y_test_original = scaler_logP.inverse_transform(y_test.values.reshape(-1, 1)).flatten() 

# Bewertung der Modellgüte mit verschiedenen Metriken
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

metrics = {
    'R²': r2_score(y_test_original, y_pred),
    'MSE': mean_squared_error(y_test_original, y_pred),
    'MAE': mean_absolute_error(y_test_original, y_pred),
}
print("Test-Metriken :", metrics)

# 12. Speichern
save_dir = "./JR_important/"

with open(save_dir + 'y_test_original_unit.pkl', 'wb') as f:
    pickle.dump(y_test_original, f)                                   
with open(save_dir + 'y_pred_unit.pkl', 'wb') as f:
    pickle.dump(y_pred, f)                                            
with open(save_dir + 'y_pred_scaled_unit.pkl', 'wb') as f:
    pickle.dump(y_pred_scaled, f)                                     
with open(save_dir + 'scaler_logP_unit.pkl', 'wb') as f:
    pickle.dump(scaler_logP, f)
with open(save_dir + 'scaler_nmx_unit.pkl', 'wb') as f:    
    pickle.dump(scaler_nmx, f)

print("Pipeline completed!")
