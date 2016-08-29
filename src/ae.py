import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras import backend as K
from keras import objectives

from mnist_loader import load_data_wrapper

results_dir = '../results/ae'

batch_size = 100
image_size = 784
h_dim = 256
latent_dim = 2

data = load_data_wrapper()
X_train, y_train = [np.hstack(a).transpose() for a in zip(*data[0])]
X_val, y_val = [np.hstack(a).transpose() for a in zip(*data[1])]
X_test, y_test= [np.hstack(a).transpose() for a in zip(*data[2])]

print(X_train.shape)
print(y_train.shape)

x = Input(batch_shape=(batch_size, image_size))
h = Dense(h_dim, activation='relu')(x)
z = Dense(latent_dim)(h)

decoder_h = Dense(h_dim, activation='relu')
h_decoded = decoder_h(z)
decoder_output = Dense(image_size, activation='sigmoid')
output = decoder_output(h_decoded)

def cost(x, output):
    cost_ce = objectives.binary_crossentropy(x, output) * image_size # Keras example multiplies with image_size
    return cost_ce

vae = Model(x, output, name='ae')
vae.compile(optimizer='rmsprop', loss=cost)

nb_epoch = 10
vae.fit(
    X_train,
    X_train,
    shuffle=True,
    nb_epoch=nb_epoch,
    batch_size=batch_size,)
#ivalidation_data=(X_val, y_val))

encoder = Model(x, z)

x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(
    x_test_encoded[:, 0],
    x_test_encoded[:, 1],
    c=np.argmax(y_test, axis=1)
)
plt.colorbar()
plt.savefig('{}/encoding_{}.png'.format(results_dir, nb_epoch))

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
x_decoded = decoder_output(_h_decoded)

generator = Model(decoder_input, x_decoded)
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded_ = generator.predict(z_sample)
        digit = x_decoded_[0].reshape(digit_size, digit_size)
        figure[
        i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size
        ] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('{}/decoding_{}.png'.format(results_dir, nb_epoch))
