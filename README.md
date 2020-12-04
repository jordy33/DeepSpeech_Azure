# DeepSpeech Spanish (Español) in Azure VM with Nvidia K80

Azure Type
```
Standard NC6_Promo (6 vcpus, 56 GiB memory) with 
Main Disk   : Standard SSD 30 Gb
Storage disk: Standard SSD 256Gb mounted in /mnt
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

Train Experimental Model in Spanish
```
cd /mnt/DeepSpeech
mkdir model
git clone https://github.com/jordy33/deepspeech_train_spanish.git es
python3 DeepSpeech.py --train_files ./es/train.csv --dev_files ./es/dev.csv --test_files ./es/test.csv --export_dir ./model
mv es old_es
```

Dowload Mozilla Common Spanish
* Get the link from the portal with click-right after entering the email and use the link between quotes in wget

```
sudo apt-get install wget
wget -O dataset.tar.gz 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-5.1-2020-06-22/es.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3OD5FZB4S%2F20201204%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201204T191504Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEGQaDD1PpBjAKH%2BGMSiaciKSBENDlO5jr9AJKO91dFdrkTf1eOtrun0TdQF%2B6fywCQ7YN674B34WRYKXWANHGZhu24HhF5sggRy6UksEQKHLWuXhSVfDJheX3hNsmpqiIHRfdH%2BIFK58Yo8rZXJ0F3BZNi%2B5Kr%2BQmBFRnEdvvPAs55N%2BlHgvwv0P1ddFiUHMZ58RWoexgE935w8AthPb%2Fg3qr1btlg7IOQj%2FVQg3u%2BMMkn0aBjVgR8GIpf4f1jUiGI1X5BTc0DDkqjldLptIhdqZtAxyVP4lwJOdvsDtZC%2B%2B%2BTAvZi%2B6GrZT%2BXEbNko%2FPn7VeBTIg2OkKQC4gs2q%2Bz5NkU5UJc95FJuhcJQ3BpX%2FvFwfxYcwW7IXB%2FYJNKbzkqh4C5ihVZtrpKWc5dhl0Su5%2BUmfnGGY9pDLMl6C9kCWjOMMs6waX7BgI6x%2FY%2B2Q5hiiM0nc17lFGavCPdL%2Fg03q7%2BmxNO%2FQvX%2Fp0y0%2FhbNxPHNKtAfWfrK5sJQdgM9gZLpRkMSiy8j1hEWM2djsUFUM%2BCrNKejj7aOmZBGPuKHG8R2qAOGe1u9eaWm663dfZ8m4Sl8ZSU2sFCbo7f%2BaL%2FL3MudGJlA8Oq4dpXvQfNbQNDZLc%2FKMh3ZHfuoTHWWfsgpNvzLK5ga3UV3xjqNOKPMbapDXoBRzySWHrbXDEkdoifFt%2FntUFgCwmtfwtqMI8O0Rt435s5QKkYIMRMxM9bKOfZU2KMWMqv4FMirL2tnsxQ%2BUA3pMLWRLtJAfC69x%2B0OOqYKVYtsfiQdJIouYXPh%2FumMV9Yk%3D&X-Amz-Signature=672860b05a272a55b0a1ec63820cb582e678bce3c333fcab4c330141f91898eb&X-Amz-SignedHeaders=host'
```

Untar 

```
tar -xaf ./dataset.tar.gz
cd dataset/cv-corpus-*
mv es /mnt/DeepSpeech/.
cd /mnt/DeepSpeech/
rm -fr ./dataset
```

Convert mp3/tsv to wav/csv
```
sudo apt install sox
sudo apt-get install libsox-fmt-mp3
bin/import_cv2.py ./es --validate_label_locale tests/test_data/validate_locale_fra.py
```
