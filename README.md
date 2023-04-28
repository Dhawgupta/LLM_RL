# LLM_RL

## installation

### **1. pull from github**

``` bash
git clone https://github.com/Sea-Snell/LLM_RL.git
cd LLM_RL
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install -e .
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
python -m pip install -e .
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip
python -m pip install jax[tpu]==0.4.8 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

### **3. install JaxSeq2**
``` shell
# navigate to a different directory
cd ~/
git clone https://github.com/Sea-Snell/JaxSeq2.git
cd JaxSeq2
python -m pip install -e .
```
