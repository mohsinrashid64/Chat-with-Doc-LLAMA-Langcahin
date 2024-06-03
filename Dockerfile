# FROM ubuntu:22.04

# # Update package lists and install necessary packages
# RUN apt update && \
#     apt install -y --no-install-recommends \
#     python3.10 \
#     python3.10-distutils \
#     curl \
#     git \
#     ca-certificates


# RUN curl -fsSL https://ollama.com/install.sh | sh

# RUN mkdir project

# WORKDIR project

# COPY . .


FROM ollama/ollama
# COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificates