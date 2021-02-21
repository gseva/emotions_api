from typing import List
from fastapi import FastAPI, File, Query

from models import PredictorResponse
from predictor import get_emotions

app = FastAPI()


@app.post('/predict/',
          response_model=PredictorResponse,
          response_model_exclude_none=True)
async def create_upload_file(
        image: bytes = File(...),
        providers: str = Query('onnx,deepai,rekognition')
        ):
    return get_emotions(image, providers.split(','))
