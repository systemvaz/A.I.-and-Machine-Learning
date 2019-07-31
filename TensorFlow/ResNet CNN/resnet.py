from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras import backend as K

import tensorflow as tf
import numpy as np
import keras
import glob
import os


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

IMG_SIZEW = 512
IMG_SIZEH = 512
IMG_CHANS = 3
NUM_CLASSES = 5
NUM_GPUS = 2
multigpu = False
n = 6

INPUT_SHAPE = (IMG_SIZEH, IMG_SIZEW, IMG_CHANS)

VERSION = 2
DEPTH = n * 9 + 2

model_type = 'ResNet%dv%d' % (DEPTH, VERSION)

#Loss rate scheduler function. Loss rate determined by epoch number
def lr_schedule(EPOCH):
    lr = 1e-3
    if EPOCH > 180:
        lr *= 0.5e-3
    elif EPOCH > 160:
        lr *= 1e-3
    elif EPOCH > 120:
        lr *= 1e-2
    elif EPOCH > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#End loss rate function

#ResNet layer
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = tf.keras.layers.Conv2D(num_filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)

    return x
#End ResNet layer

#Define our ResNet model
def resnet(input_shape, depth, num_classes=NUM_CLASSES):
    num_filters_in = 16
    num_res_blocks = int((depth -2) / 9)
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)

            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
#End of ResNet definition

#Define our model save directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'resnet_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                  cooldown=0,
                                                  patience=5,
                                                  min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

#Assign our ResNet model and define our optimiser
model = resnet(input_shape=INPUT_SHAPE, depth=DEPTH)
opt = tf.keras.optimizers.Adam

#Enable multi GPU if flagged true
if multigpu:
    model = multi_gpu_model(model, gpus=NUM_GPUS, cpu_merge=True)

#Compile and print model summary
model.compile(loss='categorical_crossentropy',
              optimizer=opt(lr=lr_schedule(0)),
              metrics=['accuracy'])

model.summary()