# STATS 507 Final Project

Umich ID: 62178151; Email: pduan@umich.edu

This repository contains the code implementation of ByteTrack and ByteTrack with Level Matching.

Due to the submission file size limit, this repository only contains the code.

If you have any question about how to run this project, please feel free to contact me.

<details>
<summary>Installation</summary>

**PyTorch with CUDA should be installed in your machine**

For Windows user and CUDA 11.X,

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install other requirements**

```shell
pip3 install -r requirements.txt
```

**Install pycocotools**
* For Ubuntu use:
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
* For Windows use:
```shell
pip3 install cython
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```


**Install YOLOX**

Clone YOLOX github repository and run setup
```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
python setup.py develop
cd ..
```


**Install FastReID**

Clone FastReID github repository
```shell
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
```

Install requirements
```shell
pip install -r docs/requirements.txt
```

Create a setup.py file with content:
```python
from setuptools import setup, find_packages

setup(name='fastreid', version='1.3', packages=find_packages(exclude=("tests", "demo", "tools")))
```

Run setup
```shell
pip install -e .
```


**Install PyQt6**

```shell
pip install PyQt6
```

</details>

<details>
<summary>How to run</summary>

```shell
python ui_main.py
```
</details>
  
