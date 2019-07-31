# TensorFlow ResNet Implementation

This is a TensorFlow implementation of a ResNet CNN for image recognition and classification. 
I have adapted it from the v2 Keras implementation at: https://keras.io/examples/cifar10_resnet/

Depth of the ResNet v2 can be defined by modifying the 'n' variable in code which corresponds to the particular model as shown in the following table: 

Model parameter
----------------------------------------------------------------------------
          |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
          |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
----------------------------------------------------------------------------
ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
---------------------------------------------------------------------------
