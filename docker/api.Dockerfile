# Model service: FastAPI plus the exported model bundle. No scikit-learn, no MLflow.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/api.txt
COPY src/ src/
COPY models/ models/
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
# PORT is set by hosting platforms such as Render; 8000 locally.
CMD ["sh", "-c", "uvicorn turbofan_rul.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
