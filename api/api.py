import fastapi as fast
from datetime import datetime
from pydantic import BaseModel
import joblib

from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("model2.pkl")

app = fast.FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    date: str
    time: str
    area: str

@app.get("/")
def read_root():
    return {"message": "what's up"}


@app.post('/predict/')
def predict(data: PredictionRequest):
    area_mapping = {
        "Wilshire": 1, "Central": 2, "Southwest": 3, "Van Nuys": 4, "Hollenbeck": 5,
        "Rampart": 6, "Newton": 7, "Northeast": 8, "77th Street": 9, "Hollywood": 10,
        "Harbor": 11, "West Valley": 12, "West LA": 13, "N Hollywood": 14, "Pacific": 15,
        "Devonshire": 16, "Mission": 17, "Southeast": 18, "Olympic": 19, "Foothill": 20,
        "Topanga": 21
    }
    area_num = area_mapping.get(data.area, 0)
    time = data.time.replace(":", "")
    month = int(data.date.split("-")[1])
    day = int(data.date.split("-")[2])

    date_obj = datetime.strptime(data.date, "%Y-%m-%d")
    dow = date_obj.weekday()



    print("inside predict")
    return {"area": area_num,
            "time": time,
            "month": month,
            "day": day,
            "dow": dow
            }
