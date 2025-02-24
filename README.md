# Eve

Eve is a **framework for building creative assistants**, leveraging open-source generative tools to provide a flexible and scalable development environment.

## Running Eve with Docker

This guide provides two options for running Eve using Docker: **a standalone container** and **a full setup with Docker Compose**.

### Option 1. Running a Single Container
-----

#### Prerequisites

* [Install Docker Compose](https://docs.docker.com/compose/install/)
* Clone this repository:

        git clone -b local-mongo --single-branch https://github.com/edenartlab/eve.git

#### Building and Running the Container

1. Navigate to the project directory

        cd eve

2. Build the Docker image

        docker build -t eden-eve .
    
3. Run the container

        docker run -d -p 8000:8000 --name eve-container eden-eve

#### Accessing the Container

        docker exec -i -t eve-container /bin/bash

### Option 2: Running Eve with MongoDB using Docker Compose
-----

For a complete setup including a **MongoDB instance**, use docker-compose to orchestrate multiple services.

Below is the `docker-compose.yml` used for this setup

```yaml
version: "3.9"

services:
  eve-server:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - PYTHON_VERSION=3.12
    container_name: eve-server
    ports:
      - "8000:8000"
    depends_on:
      - mongo
    volumes:
      - eve-data:/eve

  mongo:
    image: mongo:latest
    container_name: mongo
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=admin
    ports:
      - "27017:27017"
    restart: always
    volumes:
      - mongo-data:/data/db

volumes:
  eve-data:
    driver: local
  mongo-data:
    driver: local
```

#### Running the Full Setup

1. Ensure you have [Docker Compose installed](https://docs.docker.com/compose/install/)

2. Clone this repository (if you haven't already)

        git clone -b local-mongo --single-branch https://github.com/edenartlab/eve.git

3. Start all services

        docker-compose up -d

This will start both **Eve** and **MongoDB**, with Eve accessible at http://localhost:8000.