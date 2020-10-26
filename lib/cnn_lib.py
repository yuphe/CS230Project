import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# Convolution 2-D Layer
def Convolution2D(x, input_size, cv_filter, strides, activation='relu', wd=0):
    wshape = [cv_filter[0], cv_filter[1], cv_filter[2], cv_filter[3]]
    initializer = tf.compat.v1.keras.initializers.glorot_normal()
    w_cv = tf.get_variable("weight", shape=wshape,initializer=initializer)
    if wd>0:
      weight_decay = tf.multiply(tf.nn.l2_loss(w_cv), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    b_cv = tf.Variable(tf.constant(0.0, shape=[cv_filter[3]]),name='bias', trainable=True)

    shape_4d = [-1, input_size[0], input_size[1], cv_filter[2]]
    x_image = tf.reshape(x, shape_4d)  # reshape to 4D tensor
    z = tf.nn.conv2d(x_image, w_cv, strides=strides, padding='SAME') + b_cv

    if activation == 'relu':
        return tf.nn.relu(z)
    else:
        return z


# Max Pooling Layer
def MaxPooling2D(input, ksize, strides):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')


# Full-connected Layer
def Dense(input, n_in, n_out, wd=0):
    initializer = tf.compat.v1.keras.initializers.glorot_normal()
    w_h = tf.get_variable("weight", shape=[n_in, n_out], initializer=initializer)
    if wd>0:
      weight_decay = tf.multiply(tf.nn.l2_loss(w_h), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    b_h = tf.Variable(tf.zeros([n_out]), name='bias',trainable=True)
    logits = tf.matmul(input, w_h) + b_h
    return tf.nn.relu(logits), logits


# Read-out Layer
def ReadOutLayer(convs,input, n_in, n_out, wd=0):
    initializer = tf.compat.v1.keras.initializers.glorot_normal()
    w_o = tf.get_variable("weight", shape=[n_in, n_out], initializer=initializer)

    if wd>0:
      weight_decay = tf.multiply(tf.nn.l2_loss(w_o), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    b_o = tf.Variable(tf.zeros([n_out]),name='bias', trainable=True)
    eps = 1.e-10
    logits = tf.matmul(input, w_o) + b_o
    return convs, tf.nn.softmax(logits)+eps, logits


# Batch Normalization
def BatchNormalization(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)


# Support layer for Spatial Pooling Layer
def _max_pool_2d_nxn_regions(inputs, output_size: int, mode: str):
  """
  Performs a pooling operation that results in a fixed size:
  output_size x output_size.

  Used by spatial_pyramid_pool. Refer to appendix A in [1].

  Args:
      inputs: A 4D Tensor (B, H, W, C)
      output_size: The output size of the pooling operation.
      mode: The pooling mode {max, avg}

  Returns:
      A list of tensors, for each output bin.
      The list contains output_size * output_size elements, where
      each elment is a Tensor (N, C).

  References:
      [1] He, Kaiming et al (2015):
          Spatial Pyramid Pooling in Deep Convolutional Networks
          for Visual Recognition.
          https://arxiv.org/pdf/1406.4729.pdf.

  Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
  """
  inputs_shape = tf.shape(inputs)
  h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
  w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

  if mode == 'max':
    pooling_op = tf.reduce_max
  elif mode == 'avg':
    pooling_op = tf.reduce_mean
  else:
    msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
    raise ValueError(msg.format(mode))

  result = []
  n = output_size
  for row in range(output_size):
    for col in range(output_size):
      # start_h = floor(row / n * h)
      start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float64))), tf.int32)
      # end_h = ceil((row + 1) / n * h)
      end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float64))), tf.int32)
      # start_w = floor(col / n * w)
      start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float64))), tf.int32)
      # end_w = ceil((col + 1) / n * w)
      end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float64))), tf.int32)
      pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
      pool_result = pooling_op(pooling_region, axis=(1, 2))
      result.append(pool_result)
  return result


# Spatial Pooling Layer
def SpatialPyramidPool(inputs, dimensions=[2, 1], mode='max', implementation='kaiming'):
  """
  Performs spatial pyramid pooling (SPP) over the input.
  It will turn a 2D input of arbitrary size into an output of fixed
  dimenson.
  Hence, the convlutional part of a DNN can be connected to a dense part
  with a fixed number of nodes even if the dimensions of the input
  image are unknown.

  The pooling is performed over :math:`l` pooling levels.
  Each pooling level :math:`i` will create :math:`M_i` output features.
  :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
  of pooling operations per dimension level :math:`i`.

  The length of the parameter dimensions is the level of the spatial pyramid.

  Args:
      inputs: A 4D Tensor (N, H, W, C).
      dimensions: The list of :math:`n_i`'s that define the output dimension
      of each pooling level :math:`i`. The length of dimensions is the level of
      the spatial pyramid.
      mode: Pooling mode 'max' or 'avg'.
      implementation: The implementation to use, either 'kaiming' or 'fast'.
      kamming is the original implementation from the paper, and supports variable
      sizes of input vectors, which fast does not support.

  Returns:
      A fixed length vector representing the inputs.

  Notes:
      SPP should be inserted between the convolutional part of a DNN and it's
      dense part. Convolutions can be used for arbitrary input dimensions, but
      the size of their output will depend on their input dimensions.
      Connecting the output of the convolutional to the dense part then
      usually demands us to fix the dimensons of the network's input.
      The spatial pyramid pooling layer, however, allows us to leave
      the network input dimensions arbitrary.
      The advantage over a global pooling layer is the added robustness
      against object deformations due to the pooling on different scales.

  References:
      [1] He, Kaiming et al (2015):
          Spatial Pyramid Pooling in Deep Convolutional Networks
          for Visual Recognition.
          https://arxiv.org/pdf/1406.4729.pdf.

  Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
  """
  pool_list = []
  if implementation == 'kaiming':
    for pool_dim in dimensions:
      pool_list += _max_pool_2d_nxn_regions(inputs, pool_dim, mode)
  else:
    shape = inputs.get_shape().as_list()
    for d in dimensions:
      h = shape[1]
      w = shape[2]
      ph = np.ceil(h * 1.0 / d).astype(np.int32)
      pw = np.ceil(w * 1.0 / d).astype(np.int32)
      sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
      sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
      pool_result = tf.nn.max_pool(inputs,
                                   ksize=[1, ph, pw, 1],
                                   strides=[1, sh, sw, 1],
                                   padding='SAME')
      pool_list.append(tf.reshape(pool_result, [tf.shape(inputs)[0], -1]))
  return tf.concat(pool_list, 1)

