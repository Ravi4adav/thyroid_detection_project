FROM python:3.9.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
# CMD gunicorn --workers=4 --bind 0.0.0.0:3000 app:app
CMD ["python","app.py"]