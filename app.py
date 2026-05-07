import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from YOUTH_MIGRATION.constants import APP_HOST, APP_PORT
from YOUTH_MIGRATION.pipeline.prediction_pipeline import (
    MigrationData,
    MigrationClassifier
)
from YOUTH_MIGRATION.pipeline.training_pipeline import TrainingPipeline


# -----------------------------------
# FastAPI App
# -----------------------------------
app = FastAPI()


# -----------------------------------
# Static Files
# -----------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------------
# Templates
# -----------------------------------
templates = Jinja2Templates(directory="templates")


# -----------------------------------
# CORS
# -----------------------------------
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# Form Data Class
# -----------------------------------
class DataForm:
    def __init__(self, request: Request):

        self.request: Request = request

        self.age: Optional[str] = None
        self.gender: Optional[str] = None
        self.occupation: Optional[str] = None
        self.family_abroad: Optional[str] = None
        self.aware_programs: Optional[str] = None
        self.migration_trend: Optional[str] = None
        self.return_plan: Optional[str] = None
        self.family_impact: Optional[str] = None
        self.govt_support: Optional[str] = None
        self.responsibility: Optional[str] = None

        self.recommend_score: Optional[int] = None
        self.stress_score: Optional[int] = None
        self.country_count: Optional[int] = None

    async def get_migration_data(self):

        form = await self.request.form()

        self.age = form.get("age")
        self.gender = form.get("gender")
        self.occupation = form.get("occupation")
        self.family_abroad = form.get("family_abroad")
        self.aware_programs = form.get("aware_programs")
        self.migration_trend = form.get("migration_trend")
        self.return_plan = form.get("return_plan")
        self.family_impact = form.get("family_impact")
        self.govt_support = form.get("govt_support")
        self.responsibility = form.get("responsibility")

        self.recommend_score = int(form.get("recommend_score"))
        self.stress_score = int(form.get("stress_score"))
        self.country_count = int(form.get("country_count"))


@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        "youthmigration.html",
        {
            "request": request,
            "context": "Youth Migration Prediction System"
        }
    )


@app.get("/train")
async def train_route():

    try:

        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return Response("Training Pipeline Completed Successfully!")

    except Exception as e:

        return Response(f"Error Occurred: {e}")


@app.post("/")
async def predict_route(request: Request):

    try:

        form = DataForm(request)

        await form.get_migration_data()

        migration_data = MigrationData(
            age=form.age,
            gender=form.gender,
            occupation=form.occupation,
            family_abroad=form.family_abroad,
            aware_programs=form.aware_programs,
            migration_trend=form.migration_trend,
            return_plan=form.return_plan,
            family_impact=form.family_impact,
            govt_support=form.govt_support,
            responsibility=form.responsibility,
            recommend_score=form.recommend_score,
            stress_score=form.stress_score,
            country_count=form.country_count
        )

        migration_df = migration_data.get_input_dataframe()

        predictor = MigrationClassifier()

        prediction = predictor.predict(dataframe=migration_df)[0]

        if prediction == 1:
            status = "Likely To Move Abroad"
        else:
            status = "Not Likely To Move Abroad"

        return templates.TemplateResponse(
            "youthmigration.html",
            {
                "request": request,
                "context": status
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            "youthmigration.html",
            {
                "request": request,
                "context": f"Error: {e}"
            }
        )


if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )