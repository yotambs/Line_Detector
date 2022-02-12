import segmentation_models as sm
import random
import numpy as np
from tensorflow import keras
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

sm.set_framework('tf.keras')
datagen = ImageDataGenerator()
img_height = 512
img_width = 512

train_imgs_data_dir = "/home/yotam/workSpace/Line_detection/Data/train"
train_masks_data_dir = "/Data/masks_thin"
seed = 42
batch_Sz = 8
keras.backend.set_image_data_format('channels_last')

image_generator = datagen.flow_from_directory(
    directory= train_imgs_data_dir,
    seed = seed,
    color_mode='rgb',
    target_size=(img_height, img_width),
    batch_size=batch_Sz,
    class_mode= None)

mask_generator = datagen.flow_from_directory(
    directory= train_masks_data_dir,
    seed = seed,
    target_size=(img_height, img_width),
    batch_size=batch_Sz,
    class_mode= None)

train_generator = zip(image_generator,mask_generator)
val_generator   = zip(image_generator,mask_generator)
model = sm.Unet('resnet18', activation='relu')
#####################################
#   1
#####################################
# model.compile(
#     'Adam',
#     loss=sm.losses.dice_loss, #loss=sm.losses.bce_jaccard_loss,
#     metrics=[sm.metrics.iou_score],
# )
#####################################
#   2
#####################################
model.compile(
    'Adam',
    'binary_crossentropy',
     ['binary_accuracy']
)


model.summary()
callbacks_list=[callbacks.ReduceLROnPlateau(monitor='loss',factor=0.75,patience=1,verbose = 1,min_lr=1e-5),
callbacks.ModelCheckpoint(filepath='/home/yotam/workSpace/Line_detection/Scratch_detector_bce.h5',monitor='loss',save_best_only=True,verbose = 1)]
model.fit(
    train_generator,
    steps_per_epoch= 100/batch_Sz,
    epochs = 50,
    validation_data=val_generator,
    validation_steps = 2,
    callbacks=callbacks_list)
