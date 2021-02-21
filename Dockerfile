FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get install libgomp1

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY ./emotions_api /app
COPY ./assets /assets
