from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal, Annotated
import pickle
import pandas as pd

# Load the ml model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


app = FastAPI()

# pydantic model to validate input data
class UserInput(BaseModel):
    fixed_acidity: Annotated[float, Field(..., gt=0, lt=120, description="Fixed acidity of the wine")]
    volatile_acidity: Annotated[float, Field(..., gt=0, lt=120, description="Volatile acidity of the wine")]
    citric_acid: Annotated[float, Field(..., ge=0, lt=120, description="Citric acid of the wine")]
    residual_sugar: Annotated[float, Field(..., gt=0, lt=120, description="Residual sugar of the wine")]
    chlorides: Annotated[float, Field(..., gt=0, lt=120, description="Chlorides of the wine")]
    free_sulfur_dioxide: Annotated[float, Field(..., gt=0, lt=120, description="Free sulfur dioxide of the wine")]
    total_sulfur_dioxide: Annotated[float, Field(..., gt=0, lt=120, description="Total sulfur dioxide of the wine")]
    density: Annotated[float, Field(..., gt=0, lt=120, description="Density of the wine")]
    pH: Annotated[float, Field(..., gt=0, lt=120, description="pH of the wine")]
    sulphates: Annotated[float, Field(..., gt=0, lt=120, description="Sulphates of the wine")]
    alcohol: Annotated[float, Field(..., gt=0, lt=120, description="Alcohol content of the wine")]

@app.post("/predict")
def predict_quality(data: UserInput):
    input_df = pd.DataFrame([{
        "fixed_acidity": data.fixed_acidity,
        "volatile_acidity": data.volatile_acidity,
        "citric_acid": data.citric_acid,
        "residual_sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free_sulfur_dioxide": data.free_sulfur_dioxide,
        "total_sulfur_dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.pH,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol,
    }])


    prediction = int(model.predict(input_df)[0])
    return JSONResponse(status_code=200, content={"prediction": prediction})



