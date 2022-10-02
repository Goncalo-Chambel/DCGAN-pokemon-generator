import imageio
import os
import re
from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt


def get_dataset(folder, image_size, batch_size):
    dataset = preprocessing.image_dataset_from_directory(
        folder, label_mode=None, image_size=(image_size, image_size), batch_size=batch_size
    )
    # get images in the [-1, 1] range
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
    return dataset


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def generate_and_save_images(folder, model, epoch, seed, dim=(5, 5), figsize=(5, 5)):
    generated_images = model(seed)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = preprocessing.image.array_to_img((generated_images[i] + 1 / 2))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(folder + 'generated_image_epoch_%d.png' % epoch)
    plt.close()


def save_gif(folder):
    images = []
    images_folder = '/' + folder
    filenames = os.listdir(os.getcwd()+images_folder)
    filenames = sorted(filenames, key=numerical_sort)
    for i in range(len(filenames)):
        filename = filenames[i]
        images.append(imageio.imread(os.getcwd()+images_folder + filename))
    imageio.mimsave('generated_images.gif', images)