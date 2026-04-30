FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
