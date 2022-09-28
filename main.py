import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
import imageio
import glob
import time

batch_size = 32
dataset = preprocessing.image_dataset_from_directory(
    "pokemon_jpg/pokemon_jpg", label_mode=None, image_size=(64, 64), batch_size=batch_size
)
dataset = dataset.map(lambda x: x / 255.0)

for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break
# plt.show()




discriminator = Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(512, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        keras.layers.Dense(8 * 8 * 128),
        keras.layers.Reshape((8, 8, 128)),
        keras.layers.Conv2DTranspose(64 * 2, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.ReLU(),

        keras.layers.Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.ReLU(),

        keras.layers.Conv2DTranspose(64 * 8, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        keras.layers.ReLU(),

        keras.layers.Conv2D(3, kernel_size=5, padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False, activation='tanh')
    ],
    name="generator",
)
generator.summary()

noise = tf.random.normal([1, latent_dim])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0])
# plt.show()

decision = discriminator(generated_image)
print('decision', decision)
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss

def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    gradients_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc1, discriminator.trainable_variables))

    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)

    gradients_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc2, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))


seed = tf.random.normal([16, latent_dim])
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        generate_and_save_images(generator, epoch+1, seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    random_latent_vectors = tf.random.normal(shape=(3, latent_dim))
    generated_images = model(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(3):
        img = keras.preprocessing.image.array_to_img(generated_images[i])
        img.save("generated_images/generated_img_%03d_%d.png" % (epoch, i))

train(dataset, 50)
