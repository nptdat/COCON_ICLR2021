FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ curl sudo wget && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY transformers transformers
COPY schema schema
COPY utils utils
COPY mygenerate.py .
COPY demo_app.py .

CMD [ "streamlit", "run", "demo_app.py" ]
EXPOSE 8501
