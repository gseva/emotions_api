import os
import boto3


emotion_table = {
    'calm': 'neutral',
    'happy': 'happiness',
    'surprised': 'surprise',
    'sad': 'sadness',
    'angry': 'anger',
    'disgusted': 'disgust'
}


def call_rekognition_api(image_data):
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_access_key_id or not aws_secret_access_key:
        return {'error': 'AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are unset!'}

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    rekognition_client = session.client('rekognition', region_name='us-east-2')
    response = rekognition_client.detect_faces(
        Image={'Bytes': image_data},
        Attributes=['ALL']
    )

    emotions = response['FaceDetails'][0]['Emotions']
    confidences = {e['Type'].lower(): e['Confidence'] / 100 for e in emotions}
    return {emotion_table.get(k, k): v for k, v in confidences.items()}
