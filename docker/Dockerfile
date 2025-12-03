FROM fedora:43

EXPOSE 7860

RUN dnf install python3.13 uv -y

WORKDIR /app

COPY . /app

RUN uv pip install --system .

RUN useradd -m -u 1000 user
USER user

ENV NML_AI_CHAT_MODEL="huggingface:Qwen/Qwen3-30B-A3B-Instruct-2507:cheapest"
ENV NML_AI_EMBEDDING_MODEL="huggingface:BAAI/bge-m3:cheapest"
ENV RUNNING_IN_DOCKER=1

COPY --chown=user scripts /app/scripts
RUN chmod +x scripts/docker-deploy.sh

CMD ["./scripts/docker-deploy.sh"]
