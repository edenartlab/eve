# Use Python 3.12.8 image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# Install system dependencies for curl and bash
RUN apt update && \
    apt install -y curl bash \
    build-essential clang \
    && apt clean && \
    rm -rf /var/lib/apt/lists/*

# Set up user and working directory

ENV EVE_HOME=/root/eve
RUN python3 -m venv ${EVE_HOME}

ENV RYE_HOME=/root/.rye
ENV PATH=${RYE_HOME}/shims:${EVE_HOME}/bin:${PATH}

# Install Rye (needs to be done as root)
RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash
# Set the working directory to /eve
WORKDIR /eve

# Copy project files to the container
COPY . /eve

# Install dependencies using 'rye sync'
RUN rye sync

