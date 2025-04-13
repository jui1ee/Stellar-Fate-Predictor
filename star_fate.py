import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv('starfate_Dataset.csv')

# 2. Drop unnecessary columns
df = df.drop(columns=['name', 'spin'])

# 3. Impute missing values
df['progenitor_mass'] = df['progenitor_mass'].fillna(df['progenitor_mass'].mean())
df['collapse_time'] = df['collapse_time'].fillna(df['collapse_time'].mean())
df['metallicity'] = df['metallicity'].fillna(df['metallicity'].median())

# 4. Encode target
label_encoder = LabelEncoder()
df['type_encoded'] = label_encoder.fit_transform(df['type'])

# 5. Feature & Target split
X = df.drop(columns=['type', 'type_encoded'])
y = df['type_encoded']

# 6. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# 8. Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 9. Evaluate
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Save model artifacts
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model, scaler, and label encoder saved successfully.")
