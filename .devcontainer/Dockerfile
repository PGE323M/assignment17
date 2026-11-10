FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# cron includes the crontab command. The daemon starts on container startup,
# not during image build. Use Debian sources only for this OS package; leave
# the base image's unrelated third-party repository configuration untouched.
RUN apt-get -o Dir::Etc::sourceparts=- update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml* .devcontainer/noop.txt .devcontainer/conda_init.txt /tmp/conda-tmp/

RUN conda clean --all --yes \
    && if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && conda env update --name base --file /tmp/conda-tmp/environment.yml; fi \
    && cat /tmp/conda-tmp/conda_init.txt >> /home/vscode/.bashrc \
    && rm -rf /tmp/conda-tmp
