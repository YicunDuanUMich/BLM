# STATS 507 Final Project

Umich ID: 62178151; Email: pduan@umich.edu

This repository contains the code implementation of ByteTrack and ByteTrack with Level Matching. Due to the git commit file size limit, this repository only contains the code. If you have any question about how to run this project, please feel free to contact me.

<details>
<summary>Installation</summary>

(1) **PyTorch with CUDA should be installed in your machine**

For Windows user and CUDA 11.X,

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(2) **Install other requirements**

```shell
pip3 install -r requirements.txt
```

(3) **Install pycocotools**
* For Ubuntu use:
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
* For Windows use:
```shell
pip3 install cython
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```


(4) **Install YOLOX**

Clone YOLOX github repository and run setup
```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
python setup.py develop
cd ..
```


(5) **Install FastReID**

Clone FastReID github repository
```shell
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
```

Install requirements (If you can't install `faiss-gpu`, you can install `faiss-cpu`)
```shell
pip install -r docs/requirements.txt
```

Create a `setup.py` file with content:
```python
from setuptools import setup, find_packages

setup(name='fastreid', version='1.3', packages=find_packages(exclude=("tests", "demo", "tools")))
```

Run setup
```shell
pip install -e .
```

Put the model configuration `sbs_S50.yml` to `fast-reid/configs/MOT17/` folder:
```yaml
_BASE_: ../Base-SBS.yml

MODEL:
  BACKBONE:
    NAME: build_resnest_backbone

DATASETS:
  NAMES: ("MOT17",)
  TESTS: ("MOT17",)

OUTPUT_DIR: logs/MOT17/sbs_S50
```

</details>

<details>
<summary>How to run</summary>

Please first download the MOT17 dataset and pre-trained model. Place your MOT17 dataset under folder `./assets`. Download pre-trained weights for FastReID from [this link](https://umich-my.sharepoint.com/:u:/g/personal/pduan_umich_edu/EfZUy6S0HpNAqXn0mmTeNN0Bu0OstRjb81nBGg3Q23BOLg?e=sLQXAq) and YOLOX from [this link](https://umich-my.sharepoint.com/:u:/g/personal/pduan_umich_edu/ET3wopjD3sNMhBigNW1p-HoB-wqLua6GQvVbLYug0QE85Q?e=fh2Txa). Copy them to the folder `./assets/weights`. 

If you want to use the UI, please enter:

```shell
python ui_main.py
```

If you want to reproduce the experiment results, 

For ByteTrack:
```shell
python main_test_mot17.py .\assets\MOT17\train\ --save-metrics
```

For ByteTrack + ReID:
```shell
python main_test_mot17.py .\assets\MOT17\train\ --use-my-tracker --use-reid --save-metrics
```

For ByteTrack + ReID + Level Matching (BLM):
```shell
python main_test_mot17.py .\assets\MOT17\train\ --use-my-tracker --use-reid --use-level-match --save-metrics
```

Running the experiments takes time. I've put the experiment results to folder `./outputs`.

</details>
  
<details>
<summary>Acknowledge</summary>

The code in this repository is adapted from [ByteTrack](https://github.com/ifzhang/ByteTrack) and [MOT ByteTrack](https://github.com/0w1Gr3y/MOT_ByteTrack).

</details>