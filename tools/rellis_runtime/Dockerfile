# Rebuild recipe for the Python 3.10 / CUDA runtime on the original ROS image.
# The accepted 20261026 image is also recorded in environment.json for exact reuse.
FROM mktk1117/elevation_mapping_cupy@sha256:5d0824864cfe6c6addcaa67386701bfc41a01822c82e971413fc8baae9806f6b
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG MICROMAMBA_VERSION=2.9.0
ARG MICROMAMBA_SHA256=8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" -o /tmp/micromamba.tar.bz2 \
    && echo "${MICROMAMBA_SHA256}  /tmp/micromamba.tar.bz2" | sha256sum --check \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba \
    && rm /tmp/micromamba.tar.bz2
COPY conda-linux-64.lock requirements-pip.lock /tmp/emcp-locks/
RUN micromamba create -y -p /opt/emcupy-py310 -f /tmp/emcp-locks/conda-linux-64.lock \
    && /opt/emcupy-py310/bin/python -m pip install -r /tmp/emcp-locks/requirements-pip.lock \
    && /opt/emcupy-py310/bin/python -m pip check
WORKDIR /home/root/catkin_emcp
CMD ["sleep", "infinity"]
