#!/usr/bin/env python3

import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from fashion_util import load_fashion_data

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = load_fashion_data(download=False)


#######################
# Dimension of images #
#######################
img_width  = 28
img_height = 28
channels   = 1

######################
# Parms for learning #
######################
batch_size = 250
num_epochs = 200
iterations = 5          # number of iterations
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
# This method will increase the raw data by data augmentation of images.
# I just added rotation, horizontal flip and fill mode. Feel free to change
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
                X.append(a.reshape(img_height, img_width))
                y.append(y_)

        X.append(x_)
        y.append(y_)
    print('*Preprocessing completed: %i samples\n' % len(X))
    return np.array(X), tf.keras.utils.to_categorical(y)
    # return np.array(X), np.array(y)


train_images_shaped, train_labels_shaped = preprocess_data(
    train_images, train_labels,
    use_augmentation=True,
    nb_of_augmentation=nb_augmentation
)

test_images_shaped, test_labels_shaped = preprocess_data(test_images,  test_labels)


# Model definition
def create_model_seq():
    '''
    First layer, flatten image from 2-d array (28x28 pixels) to 1-d array (28*28=784 pixels)
    After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
    These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons).
    The second (and last) layer returns a logits array with length of 10.
    Each node contains a score that indicates the current image belongs to one of the 10 classes.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(img_height, img_width)))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(num_classes))

    '''
    Compile the model

    Loss function — This measures how accurate the model is during training. 
                    You want to minimize this function to "steer" the model in the right direction.
    Optimizer —This is how the model is updated based on the data it sees and its loss function.
    Metrics — Used to monitor the training and testing steps. 
              The following example uses accuracy, the fraction of the images that are correctly classified.
    '''
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())

    return model


# Run training for number of iterations by random data for train/validation.
# The best model of each iteration will be saved as hdf5 checkpoint
histories = []

for i in range(0, iterations):
    print('Running iteration: %i' % i)

    # Saving the best checkpoint for each iteration
    filepath = "fashion_mnist-%i.hdf5" % i

    X_train_, X_val_, y_train_, y_val_ = train_test_split(train_images_shaped, train_labels_shaped,
                                                          test_size=0.2, random_state=42)

    # model = create_model_cnn()
    model = create_model_seq()
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
