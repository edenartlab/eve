# Use Python 3.12 image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# Install system dependencies for curl and bash
RUN apt update && \
    apt install -y curl bash \
    build-essential clang libmagic1 \
    && apt clean && \
    rm -rf /var/lib/apt/lists/*

# Set up user and working directory

ENV RYE_HOME=/root/.rye
ENV PATH=${RYE_HOME}/shims:${PATH}

# Install Rye (needs to be done as root)
RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash
# Set the working directory to /eve
WORKDIR /eve

# Copy project files to the container
COPY . .

# Install dependencies using 'rye sync'
RUN rye sync

CMD ["rye", "run", "pytest", "-s", "tests"]

