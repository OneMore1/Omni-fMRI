From pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN apt-get -qq update && \
apt-get -qq install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
RUN /opt/conda/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
