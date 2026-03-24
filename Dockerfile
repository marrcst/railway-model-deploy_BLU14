FROM python:3.12

# install packages by conda
RUN pip install -r requirements_prod.txt
CMD ["python", "app.py"]
