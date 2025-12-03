FROM fedora:43

EXPOSE 7860

RUN dnf install python3.13 uv -y

RUN useradd -m -u 1000 user

RUN mkdir -p /app && chown -R user:user /app
WORKDIR /app

USER user

RUN pwd
RUN ls -lash

COPY --chown=user . /app

# add data from different repo
ADD --chown=user https://github.com/NeuroML/nml-ai-data.git /app/neuroml_ai/rag/data
RUN chmod -R 0755 /app/neuroml_ai/rag/data

RUN uv venv /app/nml-ai-venv
ENV VIRTUAL_ENV=/app/nml-ai-venv
ENV PATH="/app/nml-ai-venv/bin:$PATH"

RUN uv pip install .


COPY --chown=user scripts /app/scripts
RUN chmod +x scripts/docker-deploy.sh

ENV NML_AI_CHAT_MODEL="huggingface:Qwen/Qwen3-30B-A3B-Instruct-2507:cheapest"
ENV NML_AI_EMBEDDING_MODEL="huggingface:BAAI/bge-m3:cheapest"
ENV RUNNING_IN_DOCKER=1

CMD ["./scripts/docker-deploy.sh"]
