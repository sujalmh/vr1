# CMD ["python", "main.py"]
FROM python:3.10-slim

WORKDIR /app

COPY . .
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

