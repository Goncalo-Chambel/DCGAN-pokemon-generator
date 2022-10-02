from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

generator = load_model('Trained_models/Generator_epoch_500')

for i in range(50):
    plt.subplot(5, 10, i + 1)
    random_noise = tf.random.normal([1, 100])
    generated_image = generator(random_noise, training=False)
    plt.imshow((generated_image[0] + 1) / 2)
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
plt.tight_layout()
plt.show()
