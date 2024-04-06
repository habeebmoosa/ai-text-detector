FROM python:3.11.2

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y default-jdk

COPY . .

CMD ["gunicorn", "app:app"]