from nst_utils import *
from helper import *
import tensorflow as tf

# TENSORFLOW VERSION: 1.2.1


print(tf.__version__)


def main():

    style_layers = CONFIG.STYLE_LAYERS
    alpha = CONFIG.ALPHA
    beta = CONFIG.BETA
    num_iterations = CONFIG.NBR_ITERATIONS
    style_image = resize_image(CONFIG.STYLE_IMAGE)
    content_image = resize_image(CONFIG.CONTENT_IMAGE)

    model = load_vgg_model(CONFIG.VGG_MODEL)

    style_image = reshape_and_normalize_image(style_image)
    content_image = reshape_and_normalize_image(content_image)

    generated_image = generate_noise_image(content_image[0])

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    sess.run(model['input'].assign(content_image))
    sess.run(model['input'].assign(style_image))


    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out

    # COMPUTING CONTENT COST
    J_content = compute_content_cost(a_C, a_G)

    # COMPUTING STYLE COST
    J_style = compute_style_cost(sess, model, style_layers)

    # COMPUTING TOTAL COST
    J = total_cost(J_content, J_style, alpha, beta)

    # DEFINES OPTIMIZER
    optimizer = tf.train.AdamOptimizer(2.0)

    # DEFINES TRAIN STEP
    train_step = optimizer.minimize(J)

    # RUNS MODEL
    model_nn(sess, model, train_step, generated_image, J, J_content, J_style, num_iterations=num_iterations)


if __name__ == '__main__':
    main()

