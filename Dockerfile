FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]