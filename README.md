# DeepSpeech_Azure

Initial disk
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1        29G  1.4G   28G   5% /
```
In Ubuntu 18.04 installing Cuda and TensorFlow 10.1 
```
# Add NVIDIA package repositories
sudo apt update
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
Install Deep Speech
```
cd /mnt
sudo apt install python3-dev python3-pip python3-venv
python3 -m venv deepspeech-gpu-venv
source deepspeech-gpu-venv/bin/activate
pip3 install deepspeech-gpu
mkdir deepspeech
cd deepspeech
# Download pre-trained English model files
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/deepspeech-0.9.2-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/deepspeech-0.9.2-models.scorer
# Download example audio files
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/audio-0.9.2.tar.gz
tar xvf audio-0.9.2.tar.gz
# Transcribe an audio file
deepspeech --model deepspeech-0.9.2-models.pbmm --scorer deepspeech-0.9.2-models.scorer --audio audio/2830-3980-0043.wav
```
Result from the test
```
Loaded model in 0.204s.
Loading scorer from files deepspeech-0.9.2-models.scorer
Loaded scorer in 0.000216s.
Running inference.
experience proves this
Inference took 1.502s for 1.975s audio file.
```

Prerequisites for training own Model (prerequisite CUDA 10.0)
```
deactivate
cd /mnt
# Installing CUDA 10.0
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.5.32-1+cuda10.0  \
    libcudnn7-dev=7.6.5.32-1+cuda10.0

python3 -m venv deepspeech-train-venv
source deepspeech-train-venv/bin/activate
pip install --upgrade pip
pip3 install --upgrade pip==20.2.2 wheel==0.34.2 setuptools==49.6.0
git clone --branch v0.9.2 https://github.com/mozilla/DeepSpeech
cd DeepSpeech
pip3 install numpy==1.16.0
pip3 install --upgrade -e .
```

Train Model in Spanish
```
cd /mnt/DeepSpeech
mkdir model
git clone https://github.com/jordy33/deepspeech_train_spanish.git es
python3 DeepSpeech.py --train_files ./es/train.csv --dev_files ./es/dev.csv --test_files ./es/test.csv --export_dir ./model
```
