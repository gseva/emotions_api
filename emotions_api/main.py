
from fastapi import FastAPI, File, Query

from models import PredictorResponse
from predictor import get_emotions, EmotionSelection

app = FastAPI()


@app.post('/predict/',
          response_model=PredictorResponse,
          response_model_exclude_none=True)
async def create_upload_file(
        image: bytes = File(...),
        providers: str = Query('onnx,deepai,rekognition'),
        selection_method: EmotionSelection = Query(EmotionSelection.sum_of_scores)
        ):
    return get_emotions(image, providers.split(','), selection_method)
