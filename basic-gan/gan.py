from pathlib import Path
import time

import matplotlib.pyplot as plt

import tensorflow as tf
# TensforFlow 2.8 breaks the ability for tools (not just vs code) to find some submodules
# like keras. So I'm importing from keras directly to get rid of red squiggles
from keras import activations, layers, losses
from keras.datasets import mnist as dataset_mnist
from keras.optimizers import adam_v2 as optimizers  # only get Adam this way :'(
# older/working tensorflow:
# from tf.keras import layers, losses, datasets, optimizers


def get_dataset(batch_size):
    # don't need test images or labels
    (training_images, _), (_, _) = dataset_mnist.load_data()
    num_images = training_images.shape[0]

    # images should be 28x28, we need 28x28x1
    training_images = training_images.reshape(num_images, 28, 28, 1).astype("float32")

    # normalize to [-1, 1] ...  why tho?
    training_images = (training_images - (255 / 2)) / (255 / 2)

    return (
        tf.data.Dataset.from_tensor_slices(training_images)
            .shuffle(num_images)
            .batch(batch_size)
    )


def create_generator(noise_dim, leaky_relu_slope=0.3):

    model = tf.keras.Sequential(
        [
            layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=leaky_relu_slope),

            layers.Reshape((7, 7, 256)),
            # output shape is 7x7x256

            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
            # output shape is 7x7x128
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=leaky_relu_slope),

            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
            # output shape is 14x14x64
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=leaky_relu_slope),

            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation=activations.tanh
            ),
        ]
    )
    assert model.output_shape == (None, 28, 28, 1)

    cross_entropy = losses.BinaryCrossentropy(from_logits=True)
    def loss(fake_output):
        # generator's goal is to have the discriminator classify fakes as 1 (real)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    return model, loss


def create_discriminator(leaky_relu_slope=0.3, dropout_rate=0.3):

    model = tf.keras.Sequential(
        [
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28, 28, 1)),
            layers.LeakyReLU(alpha=leaky_relu_slope),
            layers.Dropout(dropout_rate),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=leaky_relu_slope),
            layers.Dropout(dropout_rate),

            layers.Flatten(),
            layers.Dense(1)
        ]
    )

    cross_entropy = losses.BinaryCrossentropy(from_logits=True)
    def loss(real_output, fake_output):
        # discriminator's goal is to classify real inputs as 1 and fakes as 0
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    return model, loss


SAMPLE_ROWS = 4
SAMPLE_COLUMNS = 4
NUM_SAMPLES = SAMPLE_ROWS * SAMPLE_COLUMNS
def generate_samples(generator, input, save_file=None):
    samples = generator(input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i, sample in enumerate(samples):
        plt.subplot(SAMPLE_ROWS, SAMPLE_COLUMNS, i + 1)
        plt.imshow(sample[:,:,0] * (255 / 2) + (255 / 2), cmap="gray")
        plt.axis("off")

    if save_file:
        plt.savefig(save_file)
    plt.show()


NOISE_DIM = 100
BATCH_SIZE = 256
EPOCHS = 50
CHECKPOINT_PERIOD = 5

sample_seed = tf.random.normal([NUM_SAMPLES, NOISE_DIM])

training_dataset = get_dataset(BATCH_SIZE)

generator, generator_loss = create_generator(NOISE_DIM)
discriminator, discriminator_loss = create_discriminator()

gen_optimizer = optimizers.Adam(1e-4)
disc_optimizer = optimizers.Adam(1e-4)

sample_pics_dir = Path(".") / "training" / "sample_pics"
checkpoint_dir = Path(".") / "training" / "checkpoints"
sample_pics_dir.mkdir(parents=True)
checkpoint_dir.mkdir(parents=True)

checkpoint_prefix = Path(".") / "training" / "checkpoints" / "ckpt"
checkpoint = tf.train.Checkpoint(
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    generator=generator,
    discriminator=discriminator
)

# tf.function will "compile" this function, hopefully that's good
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))


for epoch in range(1, EPOCHS + 1):
    start = time.time()

    for image_batch in training_dataset:
        train_step(image_batch)

    if epoch % CHECKPOINT_PERIOD == 0:
        print("Creating checkpoint...")
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Checkpoint {checkpoint.save_counter} created.")

    print(f"Epoch {epoch} training took {time.time() - start} seconds")
    generate_samples(generator, sample_seed, sample_pics_dir / f"epoch_{epoch}.png")

