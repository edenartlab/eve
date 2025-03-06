# Build stage for installing dependencies
FROM python:3.12-bookworm AS builder

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    clang \
    libmagic1 \
    bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV RYE_HOME=/root/.rye
ENV BREW_HOME=/home/linuxbrew/.linuxbrew/bin
ENV PATH="${RYE_HOME}/shims:${BREW_HOME}:${PATH}"

# Install Rye package manager
RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash

# Install Homebrew and required packages
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
RUN brew install ffmpeg libmagic

WORKDIR /workflows
COPY workflows /workflows

# Set working directory
WORKDIR /eve

# Copy only dependency files first for better caching
COPY eve/pyproject.toml eve/requirements*.lock ./

# Initialize rye project and sync dependencies
RUN rye sync --no-lock

# Copy the rest of the project files
COPY eve/ .

# Run sync again with all files (this time it will work with the virtual env)
RUN rye sync