import tensorflow as tf
from tensorflow.keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
import imageio
from tqdm import tqdm
import time
from Diff_Augment import DiffAugment

tf.keras.utils.set_random_seed(7)
batch_size = 32
dataset = preprocessing.image_dataset_from_directory(
    "pokemon_jpg/pokemon_jpg", label_mode=None, image_size=(128, 128), batch_size=batch_size
)
dataset = dataset.map(lambda x: (x - 127.5) / 127.5)

trained_models_folder = "aux_models_test/"
generated_images_folder = "aux_images_128_test/"
'''
for x in dataset:
    plt.axis("off")
    plt.figure(1)
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    plt.figure(2)
    plt.imshow((DiffAugment(x.numpy(), policy='color,translation') * 255)[0])
    break
plt.show()

'''

discriminator = Sequential(
    [
        tf.keras.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(1024, kernel_size=(5, 5), strides=(2, 2), padding="same",
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                               use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 100

generator = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(8 * 8 * 1024),
        tf.keras.layers.Reshape((8, 8, 1024)),
        tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                        use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(3,  kernel_size=(5, 5), strides=(1, 1), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False, activation='tanh')
    ],
    name="generator",
)
generator.summary()

random_noise = tf.random.normal([1, latent_dim])
generated_image = generator(random_noise, training=False)
# plt.imshow(generated_image[0])
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

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    # plt.subplot(1, 2, 1)
    # plt.imshow((images[0].numpy()*255).astype("int32"))
    images = DiffAugment(images, policy='color,translation,cutout')
    # plt.subplot(1, 2, 2)
    # plt.imshow((images[0].numpy()*255).astype("int32"))
    # plt.show()

    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)
        generated_images = DiffAugment(generated_images,policy='color,translation,cutout')

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
        generated_images = DiffAugment(generated_images, policy='color,translation,cutout')
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return disc_loss1 + disc_loss2, gen_loss


seed = tf.random.normal([25, latent_dim])
disc_losses = []
gen_losses = []
def train(dataset, epochs):
    generate_and_save_images(generator, 0, seed)
    discriminator.save(trained_models_folder + "Discriminator_epoch_0")
    generator.save(trained_models_folder + "Generator_epoch_0")
    for epoch in range(epochs):
        disc_loss = gen_loss = 0
        print('Currently training on epoch {} (out of {}).'.format(epoch+1, epochs))
        for image_batch in tqdm(dataset):
            losses = train_step(image_batch)
            disc_loss += losses[0]
            gen_loss += losses[1]

        generate_and_save_images(generator, epoch+1, seed)
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())

        if epoch % 100 == 0:
            discriminator.save(trained_models_folder + "Discriminator_epoch_%d" % epoch)
            generator.save(trained_models_folder + "Generator_epoch_%d" % epoch)

    generate_and_save_images(generator, epochs, seed)
    discriminator.save(trained_models_folder + "Discriminator_epoch_%d" % epochs)
    generator.save(trained_models_folder + "Generator_epoch_%d" % epochs)


def generate_and_save_images(model, epoch, seed, dim =(5, 5), figsize=(5, 5)):
    generated_images = model(seed)
    generated_images *= 255
    generated_images.numpy()
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(generated_images_folder + 'generated_image_epoch_%d.png' % epoch)
    plt.close()


train(dataset, 500)

plt.figure()
plt.plot(disc_losses, label='Discriminator Loss')
plt.plot(gen_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(generated_images_folder + 'losses.png')
plt.close()

