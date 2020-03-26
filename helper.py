from nst_utils import *
from helper import *
import tensorflow as tf


def model_nn(sess, model, train_step, input_image, J, J_content, J_style, num_iterations):
    """
    runs the whole model and save images every 20 iterations
    Returns
    The generated image
    """
    # initializes global variables
    sess.run(tf.global_variables_initializer())

    # runs the noisy input image through the model
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration: " + str(i) + " :")
            print("Total cost: = " + str(Jt))
            print("Content cost: = " + str(Jc))
            print("Style cost: = " + str(Js))

            save_image("Output/" + str(i) + ".png", generated_image)

    save_image('Output/generated_image.jpg', generated_image)

    return generated_image


def compute_content_cost(a_C, a_G):
    """
    Arguments:
    a_C - tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing the content of the image C
    a_G -tensor of dimension(1, n_H, n_W, n_C) hidden layer activations representing content of the image G

    Returns:
    J_content- content cost (float)
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


def gram_matrix(A):
    """
    argument:
    A - matrix of shape (n_C, n_H*n_W)

    Returns:
    GA - Gram matrix of A,shape (n_C, n_C)
    """

    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Argument:
    a_S -tensor of dimension (1, n_H, n_W, n_C),hidden layer activations representing style of the image S
    a_G -tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    Returns:
    J_style_layer- style cost (floaè)
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, ([n_H * n_W, n_C])))
    a_G = tf.transpose(tf.reshape(a_G, ([n_H * n_W, n_C])))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1. / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.reduce_sum(tf.pow((GS - GG), 2))

    return J_style_layer


def compute_style_cost(sess, model, style_layer):
    """
    computes the global style cost from several chosen layers
    Arguments:
    model - the tensorflow model
    style_layer - list with the names of the layers we'd like to extract the style from, and a coefficient for each of them
    Returns:
    J_style -  style cost (float)
    """

    J_style = 0

    for layer_name, coeff in style_layer:
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha, beta):
    """
    computes the total cost function

    Arguments:
    J_content - content cost
    J_style -style cost
    alpha - hyperparameter weighting the content cost
    beta - hyperparameter weighting the style cost
    Returns:
    J - total cost
    """

    J = alpha * J_content + beta * J_style

    return J