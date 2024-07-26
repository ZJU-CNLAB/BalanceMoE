# An Efficient Dynamic Load Balance Framework to Accelerate Mixture-of-Expert Training #  
## Introduction ##
This repository contains the codes of the BalanceMoE paper submitted to IEEE INFOCOM. BalanceMoE is a dynamic load balance framework to accelerate Mixture-of-Expert training. This version of BalanceMoE is implemented based on the PyTorch and FastMoE frameworks.  

## Installation ##
### Prerequisites ###
PyTorch with CUDA is required. The repository is currently tested with PyTorch v1.10.0 and CUDA 11.3.  
If the distributed expert feature is enabled, NCCL with P2P communication support, typically versions >=2.9.9.1, is needed.  
### Installing ###
Use `python setup.py install` to install BalanceMoE for training. A more detailed installation procedure can be found in [FastMoE](https://github.com/laekov/fastmoe).  
### Quick Start ###
You can download this code to /root/code folder and run the following scripts:  
```
cd /root/code/balancemoe/examples  
torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id 0 --rdzv_backend c10d text_torch_distributed.py
```  
Assume that you have 8 GPUs on a single node and everything works well, you will see that there are 8 workers running at a single node training a DummyMoEModel using the BalanceMoE.
