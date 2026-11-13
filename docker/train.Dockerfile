# Training pipeline. The same image also runs the MLflow tracking server in docker-compose.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/train.txt
COPY src/ src/
COPY configs/ configs/
COPY data/raw/ data/raw/
CMD ["python", "-m", "turbofan_rul.train"]
