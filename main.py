#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Refence : https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
# Refence : https://github.com/darienmt/CarND-Semantic-Segmentation-P2/blob/master/main.py
# Refence : https://medium.com/intro-to-artificial-intelligence/semantic-segmentation-udaitys-self-driving-car-engineer-nanodegree-c01eb6eaf9d


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # We load the vgg from the file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Then grab the default graph.
    graph = tf.get_default_graph()

    # Then get the layer by its name.
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob_tensor, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Each regulizer should be implemented on each line of code to each layer.
    # In this step, tf can get enough
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same',
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'conv_1x1_7th_layer')

    # Upsampling step. using conv2d transpose function. Kernel is 4 and stride is 2 .
    # Layer size becomes 2x2
    x2_conv7 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides = (2, 2), padding = 'same',
             kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'x2_conv7')

    # As paper mentioned, we should do skip connection.
    # First make a 1x1 convolution of pool.
    pool_4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same',
                                   kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), name = 'pool_4_1x1')

    # Adding skip layer for next step. First skip layer.
    skip1 = tf.add(x2_conv7, pool_4_1x1, name = 'skip1')

    # At this time, layer's size becomes 2x2.
    # Continuing upsampling. from skip_7_2x2_pool4_1x1 to
    upsampled_skip1 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides = (2, 2), padding = 'same',
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='4x_conv7')

    # Get conv pool3 layer.
    # The size of pool3_1x1 is 4x4 as the pool3
    pool3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='pool3_1x1')

    # So we make a skip layer2.
    skip2 = tf.add(pool3_1x1, upsampled_skip1, name = 'skip2')

    # We should upsample with stride (8,8) as paper mentioned.
    x32_upsampled = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides = (8,8), padding = 'same',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='32x_upsampled')
    return x32_upsampled

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class. origin tensor is 4d which means batch, height, weight, channel
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = "fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate values from actual labels using cross entropy
    cross_entopy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label_reshaped[:])

    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entopy, name = "fcn_loss")

    # The model implementes this operation to fin d the weights/parameters that would yield correct pixel labels.
    # Define train operation
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name = "fcn_train_op")

    return logits, train_op, loss_op

tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_value = 0.5
    learning_rate_value = 0.0001
    sum_of_loss = 0.0
    for epoch in range(epochs):
        print("**************************")
        print("Strat epoch {} ...".format(epoch + 1))
        for image, label in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict = {input_image: image, correct_label:label,
                                                                            keep_prob: keep_prob_value, learning_rate: learning_rate_value})
            sum_of_loss += loss;
            print(loss)

        print("Epoch {} ...".format(epoch +1))
        print("Total loss = {:.3f}".format(sum_of_loss))
        print("------------------------")

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    IMAGE_SHAPE = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = '/data'
    runs_dir = './runs2'
    save_model_path = './saver/model'

    # Set parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    DROPOUT = 0.75

    # Set placeholder
    correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    tests.test_for_kitti_dataset(data_dir)


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # This is the output layer of the fcn. This is the 32x upsampling.
        layers_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Get the optimize function
        # optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # return logits, train_op, loss_op
        logits, train_op, cross_entropy = optimize(layers_output, correct_label, learning_rate, num_classes)

        #initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("Tensorflow gragh build successfully, Start training .")
        #nn_last_layer, correct_label, learning_rate, num_classes
        # TODO: Train NN using the train_nn function
        # The all of the parameters in train_nn is a tensor node for trainning. Even some are functions, this is also for construct
        # tensor graph. Such as the train_op, cross_entropy input_image, correct_label keep_prob, learning_rate, these are all tensornode.
        # These are not the specified value. The specified value will be feed when the session start to run.
        # train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)
        print("Finished!")
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
