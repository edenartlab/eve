<div align="center">
  <a href="https://www.eden.art/">
    <img src="https://dtut5r9j4w7j4.cloudfront.net/94e38e25e6320e84ecdd208c8ab30abf1a03d669e4e106a8d0986c3681363f5d.png" alt="Eden Logo" width="250" height="250" style="margin-right: 20px">
    <img src="https://dtut5r9j4w7j4.cloudfront.net/d158dc1e5c62479489c1c3d119dd211bd56ba86a127359f7476990ec9e081cba.jpg" alt="Eve Logo" width="250" height="250">
  </a>
</div>

## 🚧 Under Construction 🚧

**Note: This repository is under heavy active development. Expect frequent changes and updates as we evolve the platform.**

## 🌱 Introduction

Welcome to Eve, the core repository powering [Eden.art](https://www.eden.art/)'s creative AI assistants. This codebase enables the development of autonomous digital artists capable of understanding, creating, and iterating on visual content through natural conversation.

Eve provides the foundation for building AI agents like our flagship creative assistant [Eve](https://app.eden.art/chat/eve), which leverages a suite of open-source AI tools to transform ideas into visual art.

## 🎯 Goal

Our mission is to democratize the creation of powerful creative AI agents. Eve makes it possible for anyone to build, customize, and deploy autonomous digital artists powered by open-source AI technologies.

## 🤝 Contribute Your Tools

Join our ecosystem by contributing your own tools and workflows:

- [Eden.art Node Suite for ComfyUI](https://github.com/edenartlab/eden_comfy_pipelines) - Production-ready ComfyUI nodes that power our platform
- [Eden.art Workflows](https://github.com/edenartlab/workflows) - Our in-production ComfyUI workflow repository

We invite you to contribute workflows that will become accessible to all creative agents in Eden's ecosystem!

## Agent Deploy Wiki

[Get started with Eve](https://github.com/edenartlab/eve/wiki)

## Local FalkorDB with Docker Compose

- Run `docker compose up falkordb` to start a local FalkorDB instance (default port `6380` mapped to the container’s `6379` and web UI on `http://localhost:8380`).
- Authentication is disabled by default. Set `FALKORDB_USERNAME` and `FALKORDB_PASSWORD` in your shell only if you enable credentials on a remote instance.
- Use `redis-cli -h localhost -p 6380` (or another Redis client) to verify the database is reachable before running integrations. Override the host/UI ports with `FALKORDB_PORT` and `FALKORDB_WEB_PORT` if needed.
