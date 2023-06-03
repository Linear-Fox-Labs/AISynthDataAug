import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from model import build_generator, build_discriminator, build_gan

def train(epochs, batch_size, save_interval): 
    (X_train, _), (_, _) = mnist.load_data() 
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
 
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(X_train.shape[1:])
    gan = build_gan(generator, discriminator)
 
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
 
    for epoch in range(epochs): 
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
 
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
 
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
 
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
 
        print(f'Epoch {epoch+1}/{epochs} - D loss: {d_loss:.4f} - G loss: {g_loss:.4f}')
 
        if (epoch+1) % save_interval == 0:
            save_generated_images(generator, epoch+1)

def save_generated_images(generator, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28) 
    np.save(f'gan_generated_image_epoch_{epoch}.npy', generated_images) 
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png') 
    for i in range(generated_images.shape[0]):
        cv2.imshow(f'Generated Image {i}', generated_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()