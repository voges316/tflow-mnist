#!/usr/bin/env python3

import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from fashion_util import load_fashion_data

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = load_fashion_data(download=False)

"""
we know that the images are all pre-segmented 
(e.g. each image contains a single item of clothing), that the images all have the same square size of 28×28 pixels, 
and that the images are grayscale. Therefore, we can load the images and reshape the 
data arrays to have a single color channel
"""
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)


#######################
# Dimension of images #
#######################
img_width  = 28
img_height = 28
channels   = 1

######################
# Parms for learning #
######################
batch_size = 128
num_epochs = 200
iterations = 3          # number of iterations
nb_augmentation = 4     # defines the number of additional augmentations of one image

####################
#       Data       #
####################
fashion_classes     = {0: 'T-shirt/top',
                       1: 'Trouser',
                       2: 'Pullover',
                       3: 'Dress',
                       4: 'Coat',
                       5: 'Sandal',
                       6: 'Shirt',
                       7: 'Sneaker',
                       8: 'Bag',
                       9: 'Ankle boot'}

mnist_classes       = [i for i in range(10)]
num_classes         = 10

print("Train Samples:", len(train_images))
print("Test Samples:",  len(test_images))

# Defines the options for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=False,
    fill_mode='nearest'
)


# Data augmentation (optional)
# This method will increase the raw data by data augmentation of images. I just added rotation, horizontal flip
# and fill mode. Feel free to change
def image_augmentation(image, nb_of_augmentation):
    '''
    Generates new images bei augmentation
    image : raw image
    nb_augmentation: number of augmentations
    images: array with new images
    '''
    images = []
    image = image.reshape(1, img_height, img_width, channels)
    i = 0
    for x_batch in datagen.flow(image, batch_size=1):
        images.append(x_batch)
        i += 1
        if i >= nb_of_augmentation:
            # interrupt augmentation
            break
    return images

"""
Preprocess data

Processing of raw images:
    Scaling pixels between 0.0-1.0
    Add augmentated images
"""


def preprocess_data(images, targets, use_augmentation=False, nb_of_augmentation=1):
    """
    images: raw image
    targets: target label
    use_augmentation: True if augmentation should be used
    nb_of_augmentation: If use_augmentation=True, number of augmentations
    """
    X = []
    y = []
    for x_, y_ in zip(images, targets):

        # scaling pixels between 0.0-1.0
        x_ = x_ / 255.

        # data Augmentation
        if use_augmentation:
            argu_img = image_augmentation(x_, nb_of_augmentation)
            for a in argu_img:
                X.append(a.reshape(img_height, img_width, channels))
                y.append(y_)

        X.append(x_)
        y.append(y_)
    print('*Preprocessing completed: %i samples\n' % len(X))
    return np.array(X), tf.keras.utils.to_categorical(y)


train_images_shaped, train_labels_shaped = preprocess_data(
    train_images, train_labels,
    use_augmentation=False,
    nb_of_augmentation=nb_augmentation
)

test_images_shaped, test_labels_shaped = preprocess_data(test_images,  test_labels)


# Model definition
def create_model_cnn():
    """
    Creates a simple sequential model
    """
    cnn = tf.keras.Sequential()

    cnn.add(tf.keras.layers.InputLayer(input_shape=(img_height, img_width, channels)))

    # Normalization
    cnn.add(tf.keras.layers.BatchNormalization())

    # Conv + Maxpooling
    cnn.add(tf.keras.layers.Convolution2D(32, (3, 3), kernel_initializer='he_normal', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.25))

    # Conv + Maxpooling
    cnn.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.25))

    cnn.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.4))

    # Converting 3D feature to 1D feature Vektor
    cnn.add(tf.keras.layers.Flatten())

    # Fully Connected Layer
    cnn.add(tf.keras.layers.Dense(128, activation='relu'))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.3))

    # Normalization
    cnn.add(tf.keras.layers.BatchNormalization())

    cnn.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    cnn.compile(loss='categorical_crossentropy',
                optimizer=tf.optimizers.Adam(),
                metrics=['accuracy'])

    print(cnn.summary())

    return cnn


# Run training for number of iterations by random data for train/validation.
# The best model of each iteration will be saved as hdf5 checkpoint
histories = []

for i in range(0, iterations):
    print('Running iteration: %i' % i)

    # Saving the best checkpoint for each iteration
    filepath = "fashion_mnist-%i.hdf5" % i

    X_train_, X_val_, y_train_, y_val_ = train_test_split(train_images_shaped, train_labels_shaped,
                                                          test_size=0.2, random_state=42)

    model = create_model_cnn()
    history = model.fit(
        X_train_, y_train_,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(X_val_, y_val_),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
        ]
    )

    histories.append(history.history)


# Evaluation: Training scores for loss and accuracy for all checkpoints
def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)


print('Training: \t%0.8f loss / %0.8f acc' % (get_avg(histories, 'loss'), get_avg(histories, 'accuracy')))
print('Validation: \t%0.8f loss / %0.8f acc' % (get_avg(histories, 'val_loss'), get_avg(histories, 'val_accuracy')))

# Loss/accuracy of all models on testset
# Determine loss and accuracy of all models.
test_loss = []
test_accs = []

for i in range(0, iterations):
    model_ = tf.keras.models.load_model("fashion_mnist-%i.hdf5" % i)

    score = model_.evaluate(test_images_shaped, test_labels_shaped, verbose=0)
    test_loss.append(score[0])
    test_accs.append(score[1])

    print('Running final test with model %i: %0.4f loss / %0.4f acc' % (i, score[0], score[1]))

print('\nAverage loss / accuracy on testset: %0.4f loss / %0.5f acc' % (np.mean(test_loss), np.mean(test_accs)))
print('Standard deviation: (+-%0.4f) loss / (+-%0.4f) acc' % (np.std(test_loss), np.std(test_accs)))
