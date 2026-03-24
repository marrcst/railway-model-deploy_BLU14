FROM python:3.12

# install packages by conda
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
