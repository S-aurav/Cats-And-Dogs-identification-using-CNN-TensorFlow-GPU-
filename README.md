# Cats and Dogs identification using CNN TensorFlow (GPU)

This model aims to identify cat and dog using images, it Convolution Neural Netowrk (CNN) model built using TensorFlow

## Requirements
1. Tensorflow
2. Python version >=3.8
3. Miniconda (optional, but reccommended)

## Tensorflow setup
### Running notebook on CPU (slow training):
1. Install python
   
2. Install TensorFlow (version 2.10 reccommended)
  ```
  pip install "tensorflow<=2.11"
  ```

3. Install Image library
```
pip install Image
```

4. Install Scikitlearn library
```
pip install scipy
```

### Running notebook on GPU (fast training):

Three ways to do this
 1. Native Windows install
 2. Use DirectML plugin
 3. Use WSL

Requirements: 
```
pip install Image
pip install scipy
pip install jupyter //optional but reccommended :)
```


### Native Windows install

Requirements: 

1. Visual C++ 17,15,19
```
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
```

2. Make sure long paths are enable in Windows
```
https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing)
```
### Note: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows.

Steps: 
1. create environment with python 3.9 
```
conda create --name tf python=3.9
```
2. Install Cuda and CudNN
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
3. Upgrade pip to latest version
```
pip install --upgrade pip
```

3. Install TensorFlow
```
pip install "tensorflow<2.11" 
```

Verfiy your installation:

CPU
```
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

GPU
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


### DirectML Plugin

Requirements:
Only works with Tensorflow version >= 2.12

Steps:
Install TensorFlow DirectML Plugin
```
pip install tensorflow-directml-plugin

// If TensorFlow isn't installed, it will install automatically
```


### WSL

Using Ubuntu 22.04 LTS

Steps: 
1. Make sure you have latest version of Graphics Driver installed
2. Also, make sure you are using WSL 2, as it only works with WSL version 2
3. Verify Nvidia GPU support in WSL

```
nvidia-smi
```
4. Update pip to latest version

```
pip install --upgrade pip
```
5. Install tensorflow with GPU support (Uses CUDA)

```
pip install tensorflow[and-cuda]
```

Still having trouble in setting up?, Having different Operating System other than Windows?, follow below link for detailed guide
```
https://www.tensorflow.org/install/pip
```

## Getting started
1. Clone the repository
```
git clone 
```
2. Download the Dataset from Kaggle (Link below)
```
https://www.kaggle.com/datasets/saurav818/cat-and-dog-identification-using-tensorflow/data
```

3. Extract the dataset zip file in the same directory where you cloned the repository
4. Open the notebook (CNN.ipynb) using Jupyter notebook
5. And run each cell.

## Performance evaluation
It tool 10 min and 19s for completing training on AMD Ryzen 3 3200G (CPU) and NVIDIA RTX 3070 (GPU) using TensorFlow DirectML plugin.
