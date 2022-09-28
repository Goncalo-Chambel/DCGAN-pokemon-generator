import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
import imageio
import glob
import time

batch_size = 32
all_image_path = []
for path in os.listdir('./pokemon_jpg/pokemon_jpg'):
    if '.jpg' in path:
        all_image_path.append(os.path.join('./pokemon_jpg/pokemon_jpg', path))


training_images = [np.array((Image.open(path)).resize((64,64))) for path in all_image_path]


for i in range(len(training_images)):
    training_images[i] = ((training_images[i] - training_images[i].min()) / (255 - training_images[i].min()))

training_images = np.array(training_images)
latent_dim = 100
optimizer = Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()

    generator.add(Dense(4*4*512, input_shape=[latent_dim]))
    generator.add(Reshape([4, 4, 512]))

    generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

generator = get_generator(optimizer)
generator.summary()

noise = tf.random.normal([1, latent_dim])
generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0])
# plt.show()

def get_discriminator(optimizer):
    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=[64, 64, 3]))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.4))

    discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

discriminator = get_discriminator(optimizer)
discriminator.summary()
decision = discriminator(generated_image)
print(decision)

def get_gan_network(discriminator, random_dim, generator, optimizer):
    gan = Sequential([generator, discriminator])
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


gan = get_gan_network(discriminator, latent_dim, generator, optimizer)


def generate_and_save_images(model, epoch, test_input):
    random_latent_vectors = tf.random.normal(shape=(16, latent_dim))
    generated_images = model(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(3):
        img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
        img.save("generated_images/generated_img_%03d_%d.png" % (epoch, i))


seed = tf.random.normal([16, latent_dim])

def train(epochs):
    disc_loss = []
    gen_loss = []
    batch_count = int(training_images.shape[0] / batch_size)

    for epoch in range(epochs):
        print('-'*15, 'Epoch %d' %epoch, ''*15)

        for _ in tqdm(range(batch_count)):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            image_batch = training_images[np.random.randint(0, training_images.shape[0], size=batch_size)]

            generated_images = generator.predict_on_batch(noise)
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 1

            # Train discriminator
            discriminator.trainable = True
            current_disc_loss = discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            current_gen_loss = gan.train_on_batch(noise, y_gen)

        # print('Epoch: %d,  Loss: D = %.3f, G = %.3f' %(epoch+1, current_disc_loss, current_gen_loss))
        generate_and_save_images(generator, epoch + 1, seed)
    generate_and_save_images(generator, epochs, seed)

train(200)

