
from models import Prediction, PredictorResponse
from providers import onnx, deepai, rekognition


providers_mapping = {
    'onnx': onnx.predict_classes_for_image,
    'deepai': deepai.call_deepai_api,
    'rekognition': rekognition.call_rekognition_api
}


def get_emotions(image_data, providers=['onnx', 'deepai', 'rekognition']):
    predictions = []
    for provider in providers:
        result = providers_mapping[provider](image_data)
        if 'error' in result:
            predictions.append(fill_prediction(provider, error=result['error']))
        else:
            predictions.append(fill_prediction(provider, result))

    return PredictorResponse(
        provider_predictions=predictions,
        predominant_emotion=most_common_emotion(predictions)
    )


def fill_prediction(provider, confidences=None, error=None):
    if error:
        return Prediction(provider=provider, error=error)
    predominant_emotion = get_predominant_emotion(confidences)
    return Prediction(
        provider=provider,
        confidences=confidences,
        predominant_emotion=predominant_emotion
    )


def get_predominant_emotion(confidences):
    return max(confidences, key=confidences.get)


def most_common_emotion(predictions):
    emotions = [p.predominant_emotion for p in predictions]
    return max(set(emotions), key=emotions.count)
