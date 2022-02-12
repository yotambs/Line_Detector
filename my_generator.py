import numpy as np
import keras
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(50,50), n_channels=3,
                 n_classes=1, shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment  # augment data bool
        self.on_epoch_end()

    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [

                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
                                                      n_segments=(20, 200))),
                            # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 1.0)),
                                # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(3, 5)),
                                # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 5)),
                                # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                            # sharpen images
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                       direction=(0.0, 1.0)),
                            ])),
                            iaa.AdditiveGaussianNoise(loc=0,
                                                      scale=(0.0, 0.01 * 255),
                                                      per_channel=0.5),
                            # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.01, 0.03),
                                                  size_percent=(0.01, 0.02),
                                                  per_channel=0.2),
                            ]),
                            iaa.Invert(0.01, per_channel=True),
                            # invert color channels
                            iaa.Add((-2, 2), per_channel=0.5),
                            # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-1, 1)),
                            # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-1, 0),
                                    first=iaa.Multiply((0.9, 1.1),
                                                       per_channel=True),
                                    second=iaa.ContrastNormalization(
                                        (0.9, 1.1))
                                )
                            ]),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                                sigma=0.25)),
                            # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                            # sometimes move parts of the image around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                            ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        return  seq.augment_images(images) #

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # preprocess and augment data


        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X   = np.empty((self.batch_size, *self.dim,self.n_channels ))
        ei  = np.empty((self.batch_size), dtype=bool)
        yi  = np.empty((self.batch_size), dtype=int)
        xi  = np.empty((self.batch_size), dtype=int)
        cxi = np.empty((self.batch_size), dtype=int)
        cyi = np.empty((self.batch_size), dtype=int)
        anglei = np.empty((self.batch_size), dtype=float)
        angle_x = np.empty((self.batch_size), dtype=float)
        angle_y = np.empty((self.batch_size), dtype=float)

        labels = np.empty((self.batch_size,5), dtype=int)
        #labels = np.empty((ei.__sizeof__()+xi.__sizeof__()+yi.__sizeof__()+cxi.__sizeof__()+cyi.__sizeof__(),self.batch_size))
        #labels = labels.reshape((4, 5, 4))
        #np.empty(ei.__sizeof__(),xi.__sizeof__(),self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #img = cv2.imread(list_IDs_temp[i], cv2.IMREAD_GRAYSCALE)
            # Otsu's thresholding after Gaussian filtering
            #blur = cv2.GaussianBlur(img, (3, 3), 0)
            #ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #th3  = th3 - 128
            #th3 = np.expand_dims(th3, axis=3)
            #th3 = np.expand_dims(th3, axis=3)
            #X[i,:,:,:] =th3

            img_temp = cv2.imread(list_IDs_temp[i])
            img_temp = np.expand_dims(img_temp, axis=0)
            if self.augment == True:
                img = self.augmentor(img_temp)
            # preprocess and augment data
            X[i, :, :, :] = img
            # Store class

            curr_Data = self.labels[self.labels["images"] == list_IDs_temp[i]]
            # Reading the data
            #curr_Data = train_data[train_data["images"] == list_IDs_temp[i]]
            ellipse_classifier = curr_Data["is_ellipse"].values
            cx = curr_Data["center_x"].values #- 50
            cy = curr_Data["center_y"].values #- 50
            x  = curr_Data["axis_1"].values
            y  = curr_Data["axis_2"].values
            angle = curr_Data["angle"].values

            ei[i] = ellipse_classifier
            xi[i] = x
            yi[i] = y
            cxi[i] = cx
            cyi[i] = cy
            anglei[i] = angle
            angle_y[i] = np.sin(angle)
            angle_x[i] = np.cos(angle)
            labels[i] = ellipse_classifier,x,y,cx,cy

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return (X,{'output_ellipse_classifier': ei,
                   'output_cx': cxi,
                   'output_cy': cyi,
                   'output_x' : xi,
                   'output_y' : yi,
                   'output_angle_reg': anglei,
                   'output_angle': anglei,
                   'output_angle_x': angle_x,
                   'output_angle_y': angle_y})
        #return X, labels