FROM python:3.13.0-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY data-preparation.py .

CMD ["python", "data-preparation.py"]
