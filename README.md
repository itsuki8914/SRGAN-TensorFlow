# SRGAN-TensorFlow
Implementation of SRGAN using Tensorflow

original paper: https://arxiv.org/abs/1609.04802

## Usage 

1. Download vgg19.npy from [here.](https://github.com/machrisaa/tensorflow-vgg)

  Put vgg19.npy in the folder where convert.py and convert.py are located.
  
  like this
```
...
│
├──convert.py
├──vgg.py
├──vgg19.npy
...
```

  run convert.py.
  
```
python convert.py
```

  after running, A vgg model dedicated to tensorflow will be output.
 
 ```
 ...
│
├── convert.py
├── vgg.py
├── vgg19.npy
├── modelvgg
│     ├── checkpoint
│     ├── model.ckpt-0
│     ├── model.ckpt-0
│     └── model.ckpt-0
...
```

2. Download dataset from [DIV2K dataset.](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

  download the following four.

* Train Data Track 1 bicubic downscaling x4 (LR images)
* Train Data (HR images)
* Validation Data Track 1 bicubic downscaling x4 (LR images)
* Validation Data (HR images)

(if your PC has enough RAM, I recommend also The flickr2K data set suggested in [EDSR.](https://github.com/LimBee/NTIRE2017))

  Put downloaded datasets.
