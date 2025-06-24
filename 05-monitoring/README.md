- Run this module in the new environment `monitor` with python 3.11. 
- Install the requirements.txt in the env.
- Build and run the docker compose [refer here](https://github.com/dimzachar/mlops-zoomcamp/blob/master/notes/Week_5/docker_compose.md) in the 05-monitoring directory (The defined services are available at the defined ports in the local browser)

    - The first time to build the container use: `docker-compose up --build`
    - If inorder to load the existing docker container and load the existing grafana dashboard use: `docker-compose up`
    - To stop the docker container use: `docker-compose down`
