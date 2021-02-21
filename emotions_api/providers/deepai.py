
import requests


def call_deepai_api(image_data):
    response = requests.post(
        'https://api.deepai.org/api/facial-expression-recognition',
        files={'image': image_data},
        headers={'api-key': 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K'}
    )
    result = response.json()
    if 'output' in result:
        return {v['emotion']: v['confidence']
                for v in result['output']['expressions']}
    else:
        return {'error': result['status']}
