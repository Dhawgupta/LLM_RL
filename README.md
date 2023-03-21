# context_distill4

## installation

### **1. pull from github**

``` bash
git clone --recurse-submodules https://github.com/Sea-Snell/LLM_RL.git
cd LLM_RL
export PYTHONPATH=${PWD}/src/:${PWD}/scripts/:${PWD}/JaxSeq2/src/
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
conda install jaxlib=*=*cuda* jax==0.4.6 cuda-nvcc -c conda-forge -c nvidia
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip
python -m pip install "jax[tpu]==0.3.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Update JaxSeq2

``` shell
git submodule update --remote JaxSeq2
```
