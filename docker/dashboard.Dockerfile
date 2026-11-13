# Streamlit dashboard. Talks to the API over HTTP; never loads the model.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY requirements/dashboard.txt requirements/
RUN pip install --no-cache-dir -r requirements/dashboard.txt
COPY src/ src/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/
COPY data/raw/test_FD001.txt data/raw/RUL_FD001.txt data/raw/
USER app
EXPOSE 8501
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
