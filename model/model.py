import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("../api/lstm_patrol_model.keras")

scaler = MinMaxScaler(feature_range=(0, 1))




def predict_patrol(input_features):

    try:
        input_scaled = scaler.fit_transform(input_features)
        input_reshaped = np.reshape(input_scaled, (input_scaled.shape[0], 1, input_scaled.shape[1]))
        prediction = model.predict(input_reshaped)[0]
        needs_patrol = "SEND MORE PATROL!!!" if prediction >= 0.5 else "Patrol Frequency Remains the Same"

        return {
            "prediction_value": float(prediction),
            "result": needs_patrol
        }
    except Exception as e:
        return {"error": str(e)}
