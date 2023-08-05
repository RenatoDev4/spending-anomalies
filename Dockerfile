FROM python:3.11.0

WORKDIR /app

COPY requirements.txt .

COPY . .