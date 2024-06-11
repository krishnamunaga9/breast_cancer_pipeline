FROM python:3.8-slim

COPY app.py /app/
COPY model.pkl /app/

RUN pip install flask joblib

CMD ["python", "/app/app.py"]
