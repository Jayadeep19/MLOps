FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
RUN pip install numpy

COPY ["homework_docker.py","yellow_tripdata_2023-03.parquet", "./"]

ENTRYPOINT ["python", "homework_docker.py"]