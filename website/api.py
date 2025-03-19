from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("decision_tree_crime_desc.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = FastAPI()

crime_severity_dict = {
    'BURGLARY FROM VEHICLE': 3,
    'BATTERY - SIMPLE ASSAULT': 5,
    'VEHICLE - STOLEN': 4,
    'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)': 2,
    'THEFT PLAIN - PETTY ($950 & UNDER)': 1,
    'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT': 7,
    'TRESPASSING': 1,
    'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD': 3,
    'ROBBERY': 3,
    'SHOPLIFTING - PETTY THEFT ($950 & UNDER)': 1,
    'BURGLARY': 4,
    'INTIMATE PARTNER - SIMPLE ASSAULT': 2,
    'THEFT OF IDENTITY': 3,
    'VANDALISM - MISDEMEANOR ($399 OR UNDER)': 2,
    'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)': 3,
    'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)': 2,
    'CRIMINAL THREATS - NO WEAPON DISPLAYED': 5,
    'VIOLATION OF RESTRAINING ORDER': 4,
    'BIKE - STOLEN': 2,
    'INTIMATE PARTNER - AGGRAVATED ASSAULT': 4,
    'BRANDISH WEAPON': 6,
    'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)': 3,
    'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)': 4
}


# Define request model with all required features
class CrimeInput(BaseModel):
    area: int
    area_name: str
    time_occ: int
    vict_age: int
    weapon_used_cd: int
    status: str


@app.get("/")
def home():
    return {"message": "Crime Prediction API is running!"}


@app.post("/predict/")
def predict_crime(data: CrimeInput):
    try:
        # Validate categorical inputs
        if data.area_name not in label_encoders["AREA NAME"].classes_:
            raise HTTPException(status_code=400, detail=f"Invalid area_name: {data.area_name}")

        if data.status not in label_encoders["Status"].classes_:
            raise HTTPException(status_code=400, detail=f"Invalid status: {data.status}")

        # Encode categorical features
        area_encoded = label_encoders["AREA NAME"].transform([data.area_name])[0]
        status_encoded = label_encoders["Status"].transform([data.status])[0]

        # Convert time into category
        time_category = np.digitize(data.time_occ, bins=[0, 600, 1200, 1800, 2400], right=True)

        # Get default severity if unknown
        severity = 3  # Default severity

        # Prepare feature array (Ensure correct order: 8 features)
        input_features = np.array([[data.area, area_encoded, time_category, data.mocodes, data.vict_age,
                                    data.weapon_used_cd, status_encoded, severity]])

        # Predict crime type
        prediction = model.predict(input_features)[0]

        # Decode predicted crime
        crime_desc = label_encoders["Crm Cd Desc"].inverse_transform([prediction])[0]
        if (severity >= 3):
            res="SEND MORE PATROL!!!"
        else:
            res="Patrol Frequency Remains the Same"

        return {
            "predicted_crime_description": crime_desc,
            "severity": severity,
            "result":res


        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))