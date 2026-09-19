FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 ca-certificates git binutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/mamba

COPY environment.yml /tmp/environment.yml

RUN micromamba create -f /tmp/environment.yml -y \
    && micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/mamba/envs/frgc/bin:$PATH

WORKDIR /app

CMD ["/bin/bash"]