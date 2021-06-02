import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,Dropout
from keras import optimizers

DATADIR='D:/polynomial/line/data'

CATEGORIES = ["train", "test"]
category="train"






IMG_SIZE = 170



training_data = []

def create_training_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img))
            b, g, r = cv2.split(img_array)
            rgb_img = cv2.merge([r, g, b])

            new_array = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append(new_array)  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass



            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
#print(len(training_data))



random.shuffle(training_data)
'''for sample in training_data:
    print(sample)'''


x_train = np.array(training_data, dtype="float")

max_value = float(x_train.max())
x_train =x_train / max_value




#test

category="test"



testing_data = []

def create_testing_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img))  # convert to array
            b, g, r = cv2.split(img_array)
            rgb_img = cv2.merge([r, g, b])
            new_array = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            testing_data.append(new_array)  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass



            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_testing_data()
#print(len(testing_data))



random.shuffle(testing_data)



x_test=np.array(testing_data, dtype="float")
max_value = float(x_test.max())
x_test =x_test/ max_value
# input dimension = 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
input_dim = x_train.shape[1]
print(input_dim)

encoding_dim = 64

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
)
autoencoder.add(
    Dense(input_dim, activation='sigmoid')
)

autoencoder.summary()

input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

num_images = 4
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(256,140))

for i, image_idx in enumerate(random_test_images):
    print(image_idx)
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(170, 170,-1))





    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(8,8))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(170, 170,-1))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()