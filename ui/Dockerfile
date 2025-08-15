FROM python:3.11-slim

WORKDIR /app

COPY ui/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/ .

EXPOSE 80

CMD ["python", "app.py"]
