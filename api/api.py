from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model.model import predict_patrol
from datetime import datetime

app = FastAPI()


class CrimeInput(BaseModel):
    area: int
    date: str  # Expecting YYYY-MM-DD format
    time: str  # Expecting HH:MM format


@app.get("/")
def home():
    return {"message": "Crime Prediction API is running!"}


@app.post("/predict/")
def predict_crime(data: CrimeInput):
    try:
        # Convert date to Month, Day, and DayOfWeek
        date_obj = datetime.strptime(data.date, "%Y-%m-%d")
        month = date_obj.month
        day = date_obj.day
        day_of_week = date_obj.weekday()  # Monday = 0, Sunday = 6

        # Convert time to 24-hour HHMM format
        time_occ = int(data.time.replace(":", ""))  # Converts "18:30" -> 1830

        # Convert input data to numpy array
        input_features = np.array([[data.area, time_occ, month, day, day_of_week]])

        # Call the prediction function
        result = predict_patrol(input_features)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
