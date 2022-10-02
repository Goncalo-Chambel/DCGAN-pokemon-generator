import tensorflow as tf
from tqdm import tqdm
from Diff_Augment import DiffAugment
from utils import *
from model_128 import get_optimizer, generator_loss, discriminator_loss
import model_128
import model_64

tf.keras.utils.set_random_seed(1)

image_size = 128
batch_size = 32
trained_models_folder = "Trained_models2/"
generated_images_folder = "Generated_images2/"

# get images already grouped in batches
dataset = get_dataset("pokemon_jpg/pokemon_jpg", image_size, batch_size)

'''
# Visualize images from dataset
for batch in dataset:
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(((batch.numpy() + 1) / 2)[i])  # get in the range [0, 1] for visualizing
    break
plt.show()
'''

latent_dim = 100

if image_size == 128:
    generator = model_128.get_generator(latent_dim)
    discriminator = model_128.get_discriminator()

elif image_size == 64:
    generator = model_64.get_generator(latent_dim)
    discriminator = model_64.get_discriminator()

discriminator.summary()
generator.summary()

'''
# Test models untrained
random_noise = tf.random.normal([1, latent_dim])
generated_image = generator(random_noise, training=False)
plt.imshow((generated_image[0, :, :, 0] + 1) / 2)
plt.show()

decision = discriminator(generated_image)
print("Decision:", decision)
'''

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

gen_optimizer = disc_optimizer = get_optimizer()
generator.compile(gen_optimizer, loss=binary_cross_entropy)
discriminator.compile(disc_optimizer, loss=binary_cross_entropy)


# Visualize images from dataset with diff augmentation
for batch in dataset:
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow((DiffAugment((batch.numpy() + 1) / 2, policy='color,translation,cutout'))[i])
    break
plt.show()



@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    images = DiffAugment(images, policy='color,translation,cutout')

    with tf.GradientTape() as disc_tape1:
        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output) * 0.9
        disc_loss1 = discriminator_loss(real_targets, real_output)

    gradients_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients_disc1, discriminator.trainable_variables))

    with tf.GradientTape() as disc_tape2:
        generated_images = generator(noise, training=True)
        generated_images = DiffAugment(generated_images, policy='color,translation,cutout')

        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)

    gradients_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients_disc2, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        generated_images = DiffAugment(generated_images, policy='color,translation,cutout')
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return disc_loss1 + disc_loss2, gen_loss


seed = tf.random.normal([25, latent_dim])
disc_losses = []
gen_losses = []


def train(dataset, epochs):
    generate_and_save_images(generated_images_folder, generator, 0, seed)
    discriminator.save(trained_models_folder + "Discriminator_epoch_0")
    generator.save(trained_models_folder + "Generator_epoch_0")
    for epoch in range(epochs):
        disc_loss = gen_loss = 0
        print('Currently training on epoch {} (out of {}).'.format(epoch+1, epochs))
        for image_batch in tqdm(dataset):
            losses = train_step(image_batch)
            disc_loss += losses[0]
            gen_loss += losses[1]

        generate_and_save_images(generated_images_folder, generator, epoch+1, seed)
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())

        if epoch % 100 == 0:
            discriminator.save(trained_models_folder + "Discriminator_epoch_%d" % epoch)
            generator.save(trained_models_folder + "Generator_epoch_%d" % epoch)

    generate_and_save_images(generated_images_folder, generator, epochs, seed)
    discriminator.save(trained_models_folder + "Discriminator_epoch_%d" % epochs)
    generator.save(trained_models_folder + "Generator_epoch_%d" % epochs)


train(dataset, 500)

plt.figure()
plt.plot(disc_losses, label='Discriminator Loss')
plt.plot(gen_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Losses_during_training.png')
plt.close()

save_gif(generated_images_folder)
