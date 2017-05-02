import os
from pprint import pprint

import time
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.misc import imsave
import tensorflow as tf

base_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
)
data_dir = os.path.join(base_dir, 'data')
log_dir = os.path.join(base_dir, 'logs')
content_path = os.path.join(data_dir, 'tubingen.jpg')
shape = (1, 384, 512, 3)

content_weight = 1e3
style_weight = 1e7
tv_weight = 0


def main():

    with tf.Session() as sess:
        K.set_session(sess)

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        style_image = load_image(style_path, shape)
        style_image_tensor = tf.Variable(
            style_image,
            dtype=tf.float32
        )
        imsave(
            '{}/style.jpg'.format(results_dir),
            deprocess_image(style_image, shape)
        )

        content_image = load_image(content_path, shape)
        content_image_tensor = tf.Variable(
            content_image,
            dtype=tf.float32
        )

        imsave(
            '{}/content.jpg'.format(results_dir),
            deprocess_image(content_image, shape)
        )

        combination_image = tf.Variable(
            content_image + tf.random_normal(mean=1, stddev=0.01, shape=shape),
            dtype=tf.float32
        )
        input_tensor = tf.concat(
            values=[
                style_image_tensor,
                content_image_tensor,
                combination_image
            ],
            axis=0
        )

        model = vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor
        )

        output_dict = {l.name: l.output for l in model.layers}
        pprint(output_dict.keys())

        loss = total_loss(output_dict, combination_image)
        train_step = tf.train.AdamOptimizer(learning_rate=10.0).minimize(
            loss,
            var_list=[combination_image]
        )

        n_epochs = 1000

        merged_summary = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(
            '{}/train_{}'.format(
                log_dir,
                int(time.time())
            ),
            sess.graph
        )
        # sess.run(tf.global_variables_initializer())
        initialize_variables()
        for i in range(n_epochs):
            print('Starting epoch {}'.format(i))
            summary, loss_value, _ = sess.run(
                [merged_summary, loss, train_step]
            )
            train_writer.add_summary(summary, i)
            print(loss_value)
            if i % 100 == 0:
                img = deprocess_image(
                    sess.run(combination_image),
                    shape=shape
                )
                imsave('{}/res_iter_{}.jpg'.format(results_dir, i), img)
                img = deprocess_image(
                    sess.run(style_image_tensor),
                    shape=shape
                )
        final_image = deprocess_image(
            sess.run(combination_image),
            shape=shape
        )
    imsave('{}/res.jpg'.format(results_dir), final_image)


def load_image(path, shape):
    img = load_img(path, target_size=shape[1:3])
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    return img


def total_loss(output_dict, image):
    loss = tf.Variable(0.)
    content_loss_ = content_weight * content_loss(output_dict)
    tf.summary.scalar('content_loss', content_loss_)
    loss += content_loss_

    style_loss_ = style_weight * style_loss(output_dict)
    tf.summary.scalar('style_loss', style_loss_)
    loss += style_loss_

    tv_loss = tv_weight * total_variation_loss(image, image_shape=shape)
    tf.summary.scalar('tv_loss', tv_loss)
    loss += tv_loss
    tf.summary.scalar('total_loss', loss)

    return loss


def content_loss(output_dict):
    content_layer = 'block4_conv2'
    content_features = output_dict[content_layer][1, ...]
    combined_features = output_dict[content_layer][2, ...]

    squared_diff = tf.square(content_features - combined_features)
    content_loss_ = tf.reduce_sum(squared_diff) / 2

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
        M_l = tf.cast(style_image_layer_output.shape[0] * style_image_layer_output.shape[1], tf.float32)

        squared_gram_diff = tf.square(
            gram_matrix(style_image_layer_output) - \
            gram_matrix(combined_image_layer_output)
        )
        tf.summary.histogram('gram_{}'.format(layer), squared_gram_diff)
        style_loss_ += layer_weights[i] * tf.reduce_sum(
            squared_gram_diff
        ) / (4 * N_l**2 * M_l**2)

    return style_loss_


def total_variation_loss(image, image_shape=None):
    if image_shape:
        _, width, height, channels = image_shape
    else:
        # Try to get the image size from the tensor
        dims = image.get_shape()
        width = dims[1].value
        height = dims[2].value
        channels = dims[3].value

    tv_x_size = width * (height - 1) * channels
    tv_y_size = (width - 1) * height * channels
    return (
        tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])) / tv_x_size +
        tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :])) / tv_y_size
    )


def gram_matrix(x):
    shape = x.get_shape().as_list()
    x_ = tf.reshape(x, shape=[shape[0] * shape[1], shape[2]])
    gram_matrix_ = tf.matmul(tf.transpose(x_), x_)
    return gram_matrix_


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


def get_uninitialized_variables(variables=None, session=None):
    """
    Get uninitialized variables in a session as a list.
        Args:
            variables: list of tf.Variable. Get uninitiliazed vars
                from these. If none, gets all uinitialized vars in session.
            session: tf.Session to find uninitialized vars in. If none
                uses default session.
        Returns:
            Uninitialized variables within `variables`.
            If `variables` not specified, return all uninitialized variables.
    """
    if not session:
        session = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    init_flag = session.run(
        tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]


def initialize_variables():
    '''Initialize the internal variables the optimizer uses.
        We could do tf.global_variables_initializer().eval() to
        initialize all variables but this messes up the keras model.'''
    # Get uninitialized vars and their initializers
    uninitialized_vars = get_uninitialized_variables()
    initializers = [var.initializer for var in uninitialized_vars]

    # Print uninitialized variables
    print('Uninitialized variables:')
    print([initializer.name for initializer in initializers])

    # Initialize the variables
    _ = [initializer.run() for initializer in initializers]


if __name__ == '__main__':
    names = ['seated-nude', 'Composition-VII', 'shipwreck',
             'starry_night', 'the_scream']
    for name in names:
        style_path = os.path.join(data_dir, '{}.jpg'.format(name))
        results_dir = os.path.join(
            base_dir,
            'results',
            'style_transfer',
            '{}_style{}_content{}_tv{}'.format(
                name,
                style_weight,
                content_weight,
                tv_weight
            )
        )
        main()
