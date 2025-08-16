# Pretraining with llama factory

## Install environment

Download containier
```bash
singularity pull docker://hiyouga/llamafactory:latest
singularity overlay create --size 2048 overlay_llama_factory.img
```

Run containier
```bash
HOST_CA_CERT_PATH="/etc/ssl/certs/ca-bundle.crt" # for internet connection
CONTAINER_CA_CERT_PATH="/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem" # for internet connection
apptainer shell --nv --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} --overlay overlay_llama_factory.img llamafactory_latest.sif
```

note: need to install wandb and log in inside the container.

## Set up training recipe
edit yaml: see `gpt2_wiki_pretrain.yaml`


## Test
Run pretraining locally
```bash
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train my/gpt2_wiki_pretrain.yaml
```

## Run actual job with slurm

Setup environement variable
```bash
nano ~/.bashrc
```

then add 
```bash
export CONTAINER_PATH=<path to docker image>
export OVERLAY_PATH=<path to overlay img>
export WORK_DIR=<path to LLaMA-Factory>
export YAML_PATH=<path to yaml file>
export TRITON_CACHE_DIR=<path to triton cache>
export HOST_CA_CERT_PATH=<path to ca cert, for example "/etc/ssl/certs/ca-bundle.crt">
export CONTAINER_CA_CERT_PATH=<path to container ca cert, for example "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem">
```

adjust `pretrain_gpt2.sbatch` based on hardware

Submit slurm job
```bash
sbatch pretrain_gpt2.sbatch
```