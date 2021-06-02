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
from keras import regularizers

DATADIR='D:/polynomial/line/data'

CATEGORIES = ["train", "test"]
category="train"



path = os.path.join(DATADIR,category)  # create path to dogs and cats
for img in os.listdir(path):  # iterate over each image per dogs and cats
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
    plt.imshow(img_array, cmap='gray')  # graph it
    #plt.show()  # display!

    break  # we just want one for now so break

#print(img_array)#showing pixels value
#print(img_array.shape)



IMG_SIZE = 252

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
#plt.show()

training_data = []

def create_training_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
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

x_train= np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE,1)

max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value

print('1',x_train.shape)


#print(x_train.shape)

#test

category="test"


path = os.path.join(DATADIR,category)  # create path to dogs and cats
for img in os.listdir(path):  # iterate over each image per dogs and cats
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
    plt.imshow(img_array, cmap='gray')  # graph it
    #plt.show()  # display!

    break  # we just want one for now so break

#print(img_array)#showing pixels value
#print(img_array.shape)



IMG_SIZE = 252

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
#plt.show()

testing_data = []

def create_testing_data():
    path = os.path.join(DATADIR, category)  # create path to dogs and cats


    for img in (os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
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
'''for sample in testing_data:
    print(sample)'''

x_test= np.array(testing_data).reshape(-1, IMG_SIZE, IMG_SIZE,1)

max_value = float(x_test.max())
x_test =x_test.astype('float32') / max_value
print('2',x_test.shape)

x_train = x_train.reshape((len(x_train), 252, 252, 1))
x_test = x_test.reshape((len(x_test), 252, 252, 1))



autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))
#autoencoder.summary()
# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((32, 32, 8)))

# Decoder Layers
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
encoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=64,
                validation_data=(x_test, x_test))

num_images = 2
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(256, 80))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(252, 252))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(256, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(252, 252))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()