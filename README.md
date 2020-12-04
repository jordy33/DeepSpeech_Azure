# DeepSpeech_Azure

initial disk
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1        29G  1.4G   28G   5% /
```
Ubuntu 18.04 Cuda
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.5.32-1+cuda10.1  \
    libcudnn7-dev=7.6.5.32-1+cuda10.1


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1
```

Access external disk for the user azureuser
```
sudo chown -R azureuser /mnt
chmod o+x /mnt 
```

Remove warning dataloss
```
sudo chattr -i /mnt/DATALOSS_WARNING_README.txt
sudo rm /mnt/DATALOSS_WARNING_README.txt
```

Check Nvidia is working
```
watch -n 2 nvidia-smi
```

Result
```    
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.45.01    Driver Version: 455.45.01    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 0000BE14:00:00.0 Off |                    0 |
| N/A   39C    P0    55W / 149W |      0MiB / 11441MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
Installing nvtop
```
sudo apt install cmake libncurses5-dev libncursesw5-dev git
cd /mnt
git clone https://github.com/Syllo/nvtop.git
cd nvtop
mkdir build
cd build
cmake ..
make
sudo make install
nvtop
```

It looks like
```
 Device 0 [Tesla K80] PCIe GEN 1@16x RX: N/A TX: N/A
 GPU 324MHz  MEM 324MHz  TEMP  36°C FAN N/A% POW  27 / 149 W
 GPU[                          0%] MEM[            0.000Gi/11.173Gi]
   ┌────────────────────────────────────────────────────────────────────────┐
100│                                                                   GPU 0│
   │                                                                     MEM│
   │                                                                        │
75%│                                                                        │
   │                                                                        │
   │                                                                        │
50%│                                                                        │
   │                                                                        │
   │                                                                        │
25%│                                                                        │
   │                                                                        │
 0%│────────────────────────────────────────────────────────────────────────│
   └────────────────────────────────────────────────────────────────────────┘
```
