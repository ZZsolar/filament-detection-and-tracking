# filament-detection-and-tracking
Developing an Automated Detection, Tracking and Analysis Method for Solar Filaments Observed by CHASE via Machine Learning

We have uploaded our .ipynb and .py files related to our work. Some code snippets are accompanied by comments to aid understanding.

# Device and module 

## Device:
`GPU`: NVIDIA RTX A6000;

`CUDA Version`: 11.4.

## Module:
`python`: 3.9.0;

`numpy`: 1.26.0;

`cv2`: 4.6.0;

`scipy`: 1.7.3;

`scikit-learn`:1.3.0;

`torch`: 1.10.1+cu111.

# The introduction of each code

## `data_preparation.ipynb`
We use this to get our dataset. This includes the code `readchase` for reading CHASE Hα spectral files and the code `keams_process` for preprocessing the spectra. Within this code segment, we employ 'sklearn.cluster.KMeans' for unsupervised clustering of the spectral data. Subsequently, we apply morphological closing operation on the results of K-means to obtain our dataset.

You can get the file list of our dataset in this code, and then download the files from [Solar Science Data Center of Nanjing University](https://ssdc.nju.edu.cn/NdchaseSatellite). The file is too large to be conveniently uploaded here.

## `train_unet.py`
You can run `python train_unet.py` to train the unet model. `UNet` defines our U-Net model. The original code is from [labml.ai](https://nn.labml.ai/unet/index.html), and we have made some modifications to their model. 

In addition, the code for model prediction and evaluation is also included in it.

## `track.ipynb`
This is our code for automated filament tracking. You can find the code flow of our tracking algorithm with helpful comments for better understanding. Additionally, it includes the tracking results of August 2023.

## `feature_extraction.py`
You can run `python feature_extraction.py` to obtain the Hα line central imaging of the Chase Hα file, the detection result of filaments, the cloud model inversion results of filaments, and the results of straightening the filaments along the main axis.
