FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install .

EXPOSE 8080

CMD ["python3", "app.py"]