import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('rain_data.csv')

# Drop rows with missing target
df.dropna(subset=['RainTomorrow'], inplace=True)

# Forward-fill missing values
df.ffill(inplace=True)

# Encode categorical columns
label_cols = ['RainToday', 'RainTomorrow']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature and target selection
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday']
X = df[features]
y = df['RainTomorrow']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'rainfall_model.pkl')

# Predict
predictions = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

# Save results to CSV
df['Prediction'] = predictions
df['Prediction_Label'] = df['Prediction'].apply(lambda x: 'Yes - Rain' if x == 1 else 'No - No Rain')
df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'RainToday', 'Prediction_Label']].to_csv('rainfall_predictions.csv', index=False)

# Show first 5 predictions as sample
print("Sample Predictions:")
print(df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'RainToday', 'Prediction_Label']].head())

# Confusion matrix plot
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Rainfall Prediction Confusion Matrix")
plt.savefig("rainfall_result.png")  # Save image for PPT
plt.show()
