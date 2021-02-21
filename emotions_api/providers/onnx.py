import io
import onnx
import numpy as np

from onnx import numpy_helper
from onnxruntime import backend
from PIL import Image
from scipy.special import softmax


emotion_table = {'neutral':0, 'happiness':1, 'surprise':2, 'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 'contempt':7}


def preprocess(image_data):
    input_shape = (1, 1, 64, 64)
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((64, 64), Image.ANTIALIAS).convert('L')
    img_data = np.array(img)
    img_data = np.resize(img_data, input_shape)
    return img_data.astype(np.float32)


def to_classes(scores):
    prob = np.squeeze(softmax(scores))
    inverse_mapping = {v: k for k, v in emotion_table.items()}
    return {inverse_mapping[i]: float(prob) for i, prob in enumerate(prob)}


def array_from_pb_image(path):
    tensor = onnx.TensorProto()
    with open(path, 'rb') as f:
        tensor.ParseFromString(f.read())
    return numpy_helper.to_array(tensor)


def load_model():
    return onnx.load('../assets/emotion_ferplus/model.onnx')


def predict_classes_for_image(image_data):
    inputs = preprocess(image_data)
    model = load_model()
    outputs = list(backend.run(model, inputs))
    return to_classes(outputs)
