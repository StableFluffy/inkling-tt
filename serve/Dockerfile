# Built by serve/build_image.sh from a staged context: tt-metal/ (built tree) and code/ (git archive of one commit).
ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:47772e38d127584ac8d64b4632ebdd82e5cef574f787cdb319e8f687501535dc
FROM ${BASE_IMAGE}

RUN /opt/venv/bin/uv pip install --no-cache --python /opt/venv/bin/python \
        fastapi==0.118.0 uvicorn==0.37.0 graphviz==0.21 ml_dtypes==0.4.1 \
 && printf '%s\n' /opt/tt-metal /opt/tt-metal/ttnn /opt/tt-metal/tools /opt/tt-metal/tt_metal/tt-llk/tools/python \
        > /opt/venv/lib/python3.10/site-packages/tt_metal_paths.pth \
 && mkdir -p /cache /home/inkling && chmod 1777 /cache /home/inkling

COPY tt-metal /opt/tt-metal
COPY code/inkling_tt /opt/inkling/code/inkling_tt
COPY code/serve /opt/inkling/code/serve
COPY code/LICENSE code/NOTICE /opt/inkling/

ENV TT_METAL_HOME=/opt/tt-metal \
    TT_METAL_RUNTIME_ROOT=/opt/tt-metal \
    TT_METAL_CACHE=/cache \
    ARCH_NAME=blackhole \
    HOME=/home/inkling \
    USER=inkling \
    INKLING_HF_SD=/models/hf_sd \
    INKLING_SNAPSHOT=/models/Inkling-2-Mini \
    PYTHONPATH=/opt/tt-metal:/opt/inkling/code \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/inkling/code
EXPOSE 8000
# 0.0.0.0 is the address inside the container. Publish the port on the host loopback only
# (docker run -p 127.0.0.1:8000:8000); the API has no authentication.
# Extra arguments such as --mtp --lmax 8192 are appended to the server command.
ENTRYPOINT ["python", "-u", "-m", "serve.openai_server", "--weights", "/models/hf_sd", "--tokenizer", "/models/Inkling-2-Mini", "--host", "0.0.0.0", "--port", "8000"]

# Metadata last, so a new revision label does not invalidate the layers above.
ARG BASE_IMAGE
ARG TT_METAL_COMMIT=79951bd44b7484a136b3619e0d81f26baa8a1936
ARG INKLING_VERSION=0.1.0-dev
ARG INKLING_COMMIT=unknown
ARG SOURCE_URL=https://github.com/StableFluffy/inkling-tt
LABEL org.opencontainers.image.title="inkling-2-mini-tt" \
      org.opencontainers.image.description="Inkling-2-Mini on one Tenstorrent Blackhole p150a, OpenAI-compatible server" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.documentation="${SOURCE_URL}#readme" \
      org.opencontainers.image.version="${INKLING_VERSION}" \
      org.opencontainers.image.revision="${INKLING_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      tt-metal.commit="${TT_METAL_COMMIT}"
