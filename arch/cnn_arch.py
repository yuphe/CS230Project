import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from cnn_lib import Convolution2D, MaxPooling2D, BatchNormalization
from cnn_lib import SpatialPyramidPool, Dense, ReadOutLayer

def network_1(x, w, h, c, classes, keep_prob, phase_train, wd,SPP):

    cv_strides = [1, 1, 1, 1]
    mp_strides = [1, 2, 2, 1]
    mp_filter =  [1, 2, 2, 1]
    ftsize1 = 3
    nfilter1 =32
    ftsize2 =3
    nfilter2 = 64
    nfc1 = 1024
    sppLevel = [1,2,3,6]

    with tf.variable_scope('conv_1'):
        conv1 = Convolution2D(x, (w, h), (ftsize1,ftsize1, c, nfilter1), cv_strides, activation='none', wd=wd)
        conv1_bn = BatchNormalization(conv1, nfilter1, phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1_out = MaxPooling2D(conv1_out, mp_filter, mp_strides)

    with tf.variable_scope('conv_2'):
        conv2 = Convolution2D(pool1_out, (int(w/mp_strides[1]), int(h/mp_strides[2])),
                              (ftsize2, ftsize2,nfilter1, nfilter2),
                              cv_strides, activation='none', wd=wd)
        conv2_bn = BatchNormalization(conv2,nfilter2, phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

        pool2_out = MaxPooling2D(conv2_out, mp_filter, mp_strides)

    with tf.variable_scope('fc1'):
        if SPP == True:
            dim = sum([l*l for l in sppLevel])*nfilter2
            pool2_flat = SpatialPyramidPool(pool2_out, dimensions=sppLevel)
        else:
            dim = int(pool2_out.shape[1]*pool2_out.shape[2]*pool2_out.shape[3])
            pool2_flat = tf.reshape(pool2_out, [-1, dim])

        fc1_out, _ = Dense(pool2_flat, dim, nfc1, wd)
        fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

    return ReadOutLayer([pool1_out,pool2_out],fc1_dropped, nfc1, classes, wd)

def network_2(x, w, h, c, classes, keep_prob, phase_train, wd,SPP):

    cv_strides = [1, 1, 1, 1]
    mp_strides = [1, 2, 2, 1]
    mp_filter =  [1, 2, 2, 1]
    ftsize1 = 3
    nfilter1 =6
    ftsize2 =3
    nfilter2 = 16
    nfc1 = 120
    nfc2 = 84
    sppLevel = [1,2,3,6]

    with tf.variable_scope('conv_1'):
        conv1 = Convolution2D(x, (w, h), (ftsize1,ftsize1, c, nfilter1), cv_strides, activation='none', wd=wd)
        conv1_bn = BatchNormalization(conv1, nfilter1, phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1_out = MaxPooling2D(conv1_out, mp_filter, mp_strides)

    with tf.variable_scope('conv_2'):
        conv2 = Convolution2D(pool1_out, (int(w/mp_strides[1]), int(h/mp_strides[2])),
                              (ftsize2, ftsize2,nfilter1, nfilter2),
                              cv_strides, activation='none', wd=wd)
        conv2_bn = BatchNormalization(conv2,nfilter2, phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

        pool2_out = MaxPooling2D(conv2_out, mp_filter, mp_strides)

    with tf.variable_scope('fc1'):
        if SPP == True:
            dim = sum([l*l for l in sppLevel])*nfilter2
            pool2_flat = SpatialPyramidPool(pool2_out, dimensions=sppLevel)
        else:
            dim = int(pool2_out.shape[1]*pool2_out.shape[2]*pool2_out.shape[3])
            pool2_flat = tf.reshape(pool2_out, [-1, dim])

        fc1_out, _ = Dense(pool2_flat, dim, nfc1, wd)
        fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

    with tf.variable_scope('fc2'):

        fc2_out, _ = Dense(fc1_dropped, nfc1, nfc2, wd)
        fc2_dropped = tf.nn.dropout(fc2_out, keep_prob)


    return ReadOutLayer([pool1_out,pool2_out], fc2_dropped, nfc2, classes, wd)


def network_3(x, w, h, c, classes, keep_prob, phase_train, wd,SPP):

    cv_strides = [1, 1, 1, 1]
    mp_strides = [1, 2, 2, 1]
    mp_filter =  [1, 2, 2, 1]
    ftsize1 = 3
    nfilter1 =6
    ftsize2 =3
    nfilter2 = 16
    nfc1 = 1024
    nfc2 = 512
    nfc3 = 256
    nfc4 = 128
    nfc5 = 64
    sppLevel = [1,2,3,6]

    with tf.variable_scope('conv_1'):
        conv1 = Convolution2D(x, (w, h), (ftsize1,ftsize1, c, nfilter1), cv_strides, activation='none', wd=wd)
        conv1_bn = BatchNormalization(conv1, nfilter1, phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1_out = MaxPooling2D(conv1_out, mp_filter, mp_strides)

    with tf.variable_scope('conv_2'):
        conv2 = Convolution2D(pool1_out, (int(w/mp_strides[1]), int(h/mp_strides[2])),
                              (ftsize2, ftsize2,nfilter1, nfilter2),
                              cv_strides, activation='none', wd=wd)
        conv2_bn = BatchNormalization(conv2,nfilter2, phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

        pool2_out = MaxPooling2D(conv2_out, mp_filter, mp_strides)

    with tf.variable_scope('fc1'):
        if SPP == True:
            dim = sum([l*l for l in sppLevel])*nfilter2
            pool2_flat = SpatialPyramidPool(pool2_out, dimensions=sppLevel)
        else:
            dim = int(pool2_out.shape[1]*pool2_out.shape[2]*pool2_out.shape[3])
            pool2_flat = tf.reshape(pool2_out, [-1, dim])

        fc1_out, _ = Dense(pool2_flat, dim, nfc1, wd)
        fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

    with tf.variable_scope('fc2'):

        fc2_out, _ = Dense(fc1_dropped, nfc1, nfc2, wd)
        fc2_dropped = tf.nn.dropout(fc2_out, keep_prob)

    with tf.variable_scope('fc3'):

        fc3_out, _ = Dense(fc2_dropped, nfc2, nfc3, wd)
        fc3_dropped = tf.nn.dropout(fc3_out, keep_prob)

    with tf.variable_scope('fc4'):

        fc4_out, _ = Dense(fc3_dropped, nfc3, nfc4, wd)
        fc4_dropped = tf.nn.dropout(fc4_out, keep_prob)

    with tf.variable_scope('fc5'):

        fc5_out, _ = Dense(fc4_dropped, nfc4, nfc5, wd)
        fc5_dropped = tf.nn.dropout(fc5_out, keep_prob)


    return ReadOutLayer([pool1_out,pool2_out], fc5_dropped, nfc5, classes, wd)
