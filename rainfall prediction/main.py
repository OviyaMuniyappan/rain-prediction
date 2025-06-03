import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('rain_data.csv')

# Drop rows with missing target (if any)
df.dropna(subset=['RainTomorrow'], inplace=True)

# Forward-fill missing values
df.ffill(inplace=True)

# Encode categorical columns
label_cols = ['RainToday', 'RainTomorrow']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday']
X = df[features]
y = df['RainTomorrow']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'rainfall_model.pkl')

# Predict on entire dataset
predictions = model.predict(X)

# Output prediction for each row
for i, pred in enumerate(predictions):
    msg = "It will rain tomorrow." if pred == 1 else "It will not rain tomorrow."
    print(f"Row {i + 1}: {msg}")
