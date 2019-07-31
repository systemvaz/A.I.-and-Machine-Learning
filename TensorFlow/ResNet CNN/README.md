# TensorFlow ResNet Implementation

This is a TensorFlow implementation of a ResNet CNN for image recognition and classification. 
I have adapted it from the v2 Keras implementation at: https://keras.io/examples/cifar10_resnet/

There is additional support added for multi-GPU training if required. See the section **Key Variables** below for enabling options.

## Model Depth
Depth of the ResNet v2 can be defined by modifying the 'n' variable in the python file which corresponds to the particular model as shown in the following list:

* ResNet20 v2 	  **(n=2)**
* ResNet56 v2 	  **(n=6)**
* ResNet110 v2 	  **(n=12)**
* ResNet164 v2 	  **(n=18)**
* ResNet1001 v2 	**(n=111)**

## Key Variables
**IMG_SIZEW** width of image in pixels.

**IMG_SIZEH** height of image in pixels.

**IMG_CHANS** number of colour channels (eg. 1 for grayscale, 3 for colour).

**NUM_CLASSES** number of image classification labels.

**NUM_GPUS** only matters is 'multigpu' set to True, define the number of graphics cards to use in training.

**multigpu** enable/disable multiple graphics cards for training, either False or True.

**n** corresponds to model depth described in the previous section.
