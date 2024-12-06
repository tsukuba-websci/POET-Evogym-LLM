FROM python:3.8.16

RUN apt-get update && \
    apt-get install -y --no-install-recommends git xorg-dev \
    libglu1-mesa-dev libglew-dev cmake xauth xvfb gifsicle && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules \
    https://github.com/EvolutionGym/evogym.git /evogym

# Checkout v1.0.0
RUN cd /evogym && \
    git checkout 533b9851d2b632ca43a63f39d96b917e54c77e2b

COPY . /POET-LLM

RUN cd /POET-LLM && \
    pip install -r requirements.txt

RUN cd /evogym && \
    python setup.py install
