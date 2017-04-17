import os
from pprint import pprint

from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.misc import imsave
import tensorflow as tf

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    'data'
)
style_path = os.path.join(data_dir, 'shipwreck.jpg')
content_path = os.path.join(data_dir, 'tubingen.jpg')
results_dir = '/Users/sjosund/Programming/results/neural_style/own_implementation'


def main():
    shape = (1, 200, 300, 3)
    style_image = load_image(style_path, shape)
    imsave(
        '{}/style.jpg'.format(results_dir),
        deprocess_image(style_image, shape)
    )
    style_image_tensor = tf.Variable(
        style_image,
        dtype=tf.float32
    )
    content_image = load_image(content_path, shape)
    imsave(
        '{}/content.jpg'.format(results_dir),
        deprocess_image(content_image, shape)
    )
    content_image_tensor = tf.Variable(
        content_image,
        dtype=tf.float32
    )
    combination_image = tf.Variable(
        tf.truncated_normal(
            shape=shape,
            mean=1,
            stddev=0.5
        ),
        # content_image,
        dtype='float32'
    )
    input_tensor = tf.concat(
        values=[style_image_tensor, content_image_tensor, combination_image],
        axis=0
    )

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor
    )

    output_dict = {l.name: l.output for l in model.layers}
    pprint(output_dict.keys())

    loss = total_loss(output_dict)
    train_step = tf.train.AdamOptimizer(learning_rate=1).minimize(
        loss,
        var_list=[combination_image]
    )

    n_epochs = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            print('Starting epoch {}'.format(i))
            loss_value, _ = sess.run(
                [loss, train_step]
            )
            print(loss_value)
            if i % 10 == 0:
                img = deprocess_image(
                    sess.run(combination_image),
                    shape=shape
                )
                imsave('{}/res_iter_{}.jpg'.format(results_dir, i), img)
        final_image = deprocess_image(
            sess.run(combination_image),
            shape=shape
        )
    imsave('{}/res.jpg'.format(results_dir), final_image)


def load_image(path, shape):
    img = img_to_array(load_img(path, target_size=shape[1:3]))
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    return img


def total_loss(output_dict):
    alpha = 5e0
    beta = 1e2
    loss = tf.Variable(0.)
    loss += alpha * content_loss(output_dict)
    loss += beta * style_loss(output_dict)

    return loss


def content_loss(output_dict):
    content_layer = 'block4_conv2'
    content_features = output_dict[content_layer][1, ...]
    combined_features = output_dict[content_layer][2, ...]

    content_loss_ = tf.reduce_sum(
        tf.square(content_features - combined_features) / 2
    )

    return content_loss_


def style_loss(output_dict):
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1',
        'block5_conv1'
    ]
    layer_weights = [1/len(style_layers) for _ in range(len(style_layers))]
    style_loss_ = tf.Variable(0.)
    for i, layer in enumerate(style_layers):
        style_image_layer_output = output_dict[layer][0, ...]
        combined_image_layer_output = output_dict[layer][2, ...]

        N_l = tf.cast(style_image_layer_output.shape[2], tf.float32)
        M_l = tf.cast(
            style_image_layer_output.shape[0] * style_image_layer_output.shape[1],
            tf.float32
        )

        style_loss_ += layer_weights[i] / (4 * N_l**2 * M_l**2) * tf.reduce_sum(
            tf.square(
                gram_matrix(style_image_layer_output) - gram_matrix(combined_image_layer_output)
            )
        )

    return style_loss_


def gram_matrix(x):
    shape = x.get_shape().as_list()
    x = tf.reshape(x, shape=[shape[0] * shape[1], shape[2]])
    return tf.matmul(tf.transpose(x), x)


def deprocess_image(x, shape):
    x = x.reshape((shape[1], shape[2], 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


if __name__ == '__main__':
    main()