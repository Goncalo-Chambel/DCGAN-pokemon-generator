import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, LeakyReLU, ReLU, Reshape, Conv2DTranspose
from keras.optimizers import Adam



trained_models_folder = "aux_models_test/"
generated_images_folder = "aux_images_128_test/"
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss


def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss


def get_discriminator():
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    discriminator = Sequential()

    discriminator.add(Input(shape=(128, 128, 3)))

    # 128*128*3 -> 64*64*64
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same",
                             kernel_initializer=initializer, use_bias=False))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    discriminator.add(LeakyReLU(0.2))

    # 64x64x64 -> 32x32x128
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same",
                             kernel_initializer=initializer, use_bias=False))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    discriminator.add(LeakyReLU(0.2))

    # 32x32x128 -> 16x16x256
    discriminator.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                             kernel_initializer=initializer, use_bias=False))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    discriminator.add(LeakyReLU(0.2))

    # 16x16x256 -> 16x16x512
    discriminator.add(Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding="same",
                             kernel_initializer=initializer, use_bias=False))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    discriminator.add(LeakyReLU(0.2))

    # 16x16x512 -> 8x8x1024
    discriminator.add(Conv2D(1024, kernel_size=(5, 5), strides=(2, 2), padding="same",
                             kernel_initializer=initializer, use_bias=False))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    return discriminator


def get_generator(latent_dim):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    generator = Sequential()

    generator.add(Input(shape=(latent_dim,)))
    # 8*8*1024
    generator.add(Dense(8 * 8 * 1024))
    generator.add(Reshape((8, 8, 1024)))

    # 8x8x1024 -> 16x16x512
    generator.add(Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  kernel_initializer=initializer, use_bias=False))
    generator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    generator.add(ReLU())

    # 16x16x512 -> 32x32x256
    generator.add(Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  kernel_initializer=initializer, use_bias=False))
    generator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    generator.add(ReLU())

    # 32x32x256 -> 64x64x128
    generator.add(Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  kernel_initializer=initializer, use_bias=False))
    generator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    generator.add(ReLU())

    # 64x64x128 -> 128x128x64
    generator.add(Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  kernel_initializer=initializer, use_bias=False))
    generator.add(BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02))
    generator.add(ReLU())

    # 128x128x64 -> 128x128x3
    generator.add(Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same',
                           kernel_initializer=initializer, use_bias=False, activation='tanh'))

    return generator

def get_optimizer(lr=0.0002, beta1=0.5):
    return Adam(lr, beta1)

'''
generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)



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

'''

