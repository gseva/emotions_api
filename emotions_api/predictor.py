
from enum import Enum
from collections import defaultdict
from operator import itemgetter

from models import Prediction, PredictorResponse
from providers import onnx, deepai, rekognition


providers_mapping = {
    'onnx': onnx.predict_classes_for_image,
    'deepai': deepai.call_deepai_api,
    'rekognition': rekognition.call_rekognition_api
}


class EmotionSelection(str, Enum):
    most_common = 'most_common'
    highest_score = 'highest_score'
    sum_of_scores = 'sum_of_scores'


def get_emotions(image_data, providers=['onnx', 'deepai', 'rekognition'],
                 selection_method=EmotionSelection.most_common):
    selection_method = emotion_selectors_mapping[selection_method]

    predictions = []
    for provider in providers:
        result = providers_mapping[provider](image_data)
        if 'error' in result:
            predictions.append(fill_prediction(provider, error=result['error']))
        else:
            predictions.append(fill_prediction(provider, result))

    return PredictorResponse(
        provider_predictions=predictions,
        predominant_emotion=selection_method(predictions)
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


def emotion_with_highest_score(predictions):
    emotion, highest_score = None, 0
    for pred in predictions:
        if pred.confidences:
            for k, v in pred.confidences.items():
                if v > highest_score:
                    emotion = k
                    highest_score = v
    return emotion


def emotion_with_higher_sum_of_scores(predictions):
    scores = defaultdict(float)
    for pred in predictions:
        if pred.confidences:
            for conf, val in pred.confidences.items():
                scores[conf] += val
    return max(scores, key=scores.get)


def most_common_emotion(predictions):
    emotions = [p.predominant_emotion for p in predictions]
    return max(set(emotions), key=emotions.count)


emotion_selectors_mapping = {
    EmotionSelection.most_common: most_common_emotion,
    EmotionSelection.highest_score: emotion_with_highest_score,
    EmotionSelection.sum_of_scores: emotion_with_higher_sum_of_scores
}
