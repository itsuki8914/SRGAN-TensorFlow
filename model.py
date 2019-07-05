import tensorflow as tf
import numpy as np

REGULARIZER_COF = 1e-8

def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormc"+name)
    h = tf.nn.leaky_relu(h)
    return h

def _up_sampling(x):
    bs, h, w, c = x.get_shape().as_list()
    #h = tf.image.resize_nearest_neighbor(x,[h*2,w*2])
    #h = tf.image.resize_bicubic(x,[h*2,w*2])
    h = tf.image.resize_bilinear(x,[h*2,w*2])
    return h

def buildSRGAN_g(x,reuse=False,isTraining=True):

    with tf.variable_scope("SRGAN_g", reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        conv_w, conv_b = _conv_variable([7,7,3,64],name="conv4_g")
        h = _conv2d(x,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        tmp = h

        for i in range(16):
            conv_w, conv_b = _conv_variable([3,3,64,64],name="res%s-1" % i)
            nn = _conv2d(h,conv_w,stride=1) + conv_b
            nn = tf.contrib.layers.batch_norm(nn, decay=0.8, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm%s-1_g" %i)
            nn = tf.nn.leaky_relu(nn)
            conv_w, conv_b = _conv_variable([3,3,64,64],name="res%s-2" % i)
            nn = _conv2d(nn,conv_w,stride=1) + conv_b
            nn = tf.contrib.layers.batch_norm(nn, decay=0.8, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm%s-2_g" %i)

            nn = tf.math.add(h,nn*0.5, name="resadd%s" % i)
            h = nn

        conv_w, conv_b = _conv_variable([3,3,64,64],name="conv_3")
        nn = _conv2d(h,conv_w,stride=1) + conv_b
        nn = tf.contrib.layers.batch_norm(nn, decay=0.8, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm_3_g" )
        h = tmp + h

        h = _up_sampling(h)
        conv_w, conv_b = _conv_variable([3,3,64,64],name="conv_2")
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        h = _up_sampling(h)
        conv_w, conv_b = _conv_variable([3,3,64,64],name="conv_1")
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        conv_w, conv_b = _conv_variable([7,7,64,3],name="conv_o_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y

def buildSRGAN_d(x,reuse=False,isTraining=True):
    with tf.variable_scope("SRGAN_d", reuse=reuse) as scope:
        h = _conv_layer(x, 3, 32, 1, 3, "1-1_d", isTraining=isTraining)
        h = _conv_layer(h, 32, 32, 2, 3, "1-2_d", isTraining=isTraining)

        h = _conv_layer(h, 32, 64, 1, 3, "2-1_d", isTraining=isTraining)
        h = _conv_layer(h, 64, 64, 2, 3, "2-2_d", isTraining=isTraining)

        h = _conv_layer(h, 64, 128, 1, 3, "3-1_d", isTraining=isTraining)
        h = _conv_layer(h, 128, 128, 2, 3, "3-2_d", isTraining=isTraining)

        h = _conv_layer(h, 128, 256, 1, 3, "4-1_d", isTraining=isTraining)
        h = _conv_layer(h, 256, 256, 2, 3, "4-2_d", isTraining=isTraining)

        n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
        h = tf.reshape(h,[n_b,n_h*n_w*n_f])

        w, b = _fc_variable([n_h*n_w*n_f,512],name="fc2")
        h = tf.matmul(h, w) + b
        h = tf.nn.leaky_relu(h)

        w, b = _fc_variable([512, 1],name="fc1")
        h = tf.matmul(h, w) + b
        y = tf.nn.sigmoid(h)
    return y
