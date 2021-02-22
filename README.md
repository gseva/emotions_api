
## API de reconocimiento de emociones en imagenes de caras

Esta API devuelve la emoción predominante dada una imagen de cara. Lo hace
conectandose a diferentes proveedores, actualmente son 3:

 - onnx: hace uso del modelo preentrenado [emotions fer+](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus) de onnx. Devuelve probabilidad de las 8 emociones básicas.
 - deepai: hace uso de la [api abierta de reconocimiento de expresión facial](https://deepai.org/machine-learning-model/facial-expression-recognition) de deepai. Devuelve la emoción predominante, y la confianza de ella.
 - rekognition: hace uso del servicio [aws rekognition](https://aws.amazon.com/rekognition/). Devuelve la probabilidad de 8 emociones (a diferencia de onnx, la 8va emoción es `confused` en lugar de `contempt`)

### Como correr localmente

La API fue desarrollada y testada para Python 3.7 o más nuevo. Primero hay que
instalar las dependencias (se recomienda primero crear un entorno virtual):

```bash
pip install -r requirements.txt
```

Correr la API:

```bash
cd emotions_api
uvicorn main:app
```

Si se quiere usar el proveedor de AWS Rekognition, se necesita especificar la
clave de acceso a AWS via variables de entorno. Se puede hacer de la siguiente
manera, reemplazando los `...` por los valores de las claves:

```bash
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...  uvicorn main:app
```


### Como correr con Docker

Primero hay que crear la imágen:

```bash
docker build -t emotions-api .
```

Luego correr un contenedor con esa imagen:

```bash
docker run --name emotions-api-container \
  -p 8000:80 emotions-api
```

Para correr el contenedor pasandole las claves de AWS:

```bash
docker run --name emotions-api-container \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -p 8000:80 emotions-api
```

### Como usar la API

Una vez que la API está levantada, se pueden hacer pruebas con las imágenes que vienen en assets, u otras imágenes, usando `curl` de la siguiente manera:

```bash
curl -F "image=@assets/images/happiness.jpg" localhost:8000/predict/
```

Se puede especificar qué proveedores usar via parámetro `providers`, separados por coma.
Los valores aceptados son `onnx,deepai,rekognition`

```bash
curl -F "image=@assets/images/happiness.jpg" \
  localhost:8000/predict/?providers=onnx,rekognition
```

Respuesta ejemplo de la API:

```json
{
  "provider_predictions": [
    {
      "provider": "onnx",
      "confidences": {
        "neutral": 3.380749603820732e-06,
        "happiness": 0.999995231628418,
        "surprise": 1.2634276345124817e-06,
        "sadness": 3.720915797700286e-09,
        "anger": 1.856849074499678e-08,
        "disgust": 1.0785463633311565e-09,
        "fear": 1.243859459876262e-09,
        "contempt": 5.172450556756303e-08
      },
      "predominant_emotion": "happiness"
    },
    {
      "provider": "rekognition",
      "confidences": {
        "happiness": 98.02332305908203,
        "surprise": 0.4820031523704529,
        "fear": 0.41370725631713867,
        "anger": 0.3769048750400543,
        "disgust": 0.2959006428718567,
        "confused": 0.2704017162322998,
        "sadness": 0.08125089108943939,
        "neutral": 0.056507404893636703
      },
      "predominant_emotion": "happiness"
    }
  ],
  "predominant_emotion": "happiness"
}
```

### Documentación de la API

Una vez levantada, se puede acceder la documentación de la API entrando a `http://localhost:8000/docs`.

Se puede acceder otra versión de documentación entrando a `http://localhost:8000/redoc`.


### Correr los notebooks

Primero hay que instalar las dependencias de los notebooks (además de las dependencias normales):

```bash
pip install -r requirements.txt -r requirements_notebooks.txt
```

Luego correr

```bash
jupyter notebook
```
