# This code is released under the CC BY-SA 4.0 license.

import tensorflow as tf


# SSPCAB implementation
def sspcab_layer(input, name, kernel_dim, dilation, filters, reduction_ratio=8):
    '''
        input: The input data
        name: The name of the layer in the graph
        kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
        dilation: The dilation dimension 'd' from the paper
        filters: The number of filter at the output (usually the same with the number of filter from the input)
        reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
    '''
    with tf.compat.v1.variable_scope('SSPCAB/' + name) as scope:
        pad = kernel_dim + dilation
        border_input = kernel_dim + 2*dilation + 1
        
        
        
        sspcab_input = tf.pad(input, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), "REFLECT")
        
        sspcab_1 = tf.keras.layers.Conv2D(
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)(sspcab_input[:, :-border_input, :-border_input, :])
        sspcab_3 = tf.keras.layers.Conv2D(
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)(sspcab_input[:, border_input:, :-border_input, :])
        sspcab_7 = tf.keras.layers.Conv2D(
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)(sspcab_input[:, :-border_input, border_input:, :])
        sspcab_9 = tf.keras.layers.Conv2D(
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)(sspcab_input[:, border_input:, border_input:, :])
        
        sspcab_out = sspcab_1 + sspcab_3 + sspcab_7 + sspcab_9

        se_out = se_layer(sspcab_out, filters, reduction_ratio, 'SSPCAB/se_' + name)
        return se_out


# Squeeze and Excitation block
def se_layer(input_x, in_channels, ratio, layer_name):
    '''
        input_x: The input data
        out_dim: The number of input channels
        ration: The reduction ratio 'r' from the paper
        layer_name: The name of the layer in the graph
    '''
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, axis=[1, 2])
        excitation = tf.keras.layers.Dense(use_bias=True, units=in_channels / ratio)(squeeze)
        excitation = tf.nn.relu(excitation)
        excitation = tf.keras.layers.Dense(use_bias=True, units=in_channels)(excitation)
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, in_channels])
        scale = input_x * excitation
        return scale

# Example of how our block should be updated
# cost_sspcab = tf.square(self.input_sspcab - output_sspcab)
# loss = 0.1 * tf.reduce_mean(cost_sspcab)

class SSPCAB(tf.keras.layers.Layer):
    def __init__(self, kernel_dim, dilation, filters, reduction_ratio=8, name=None):
        super(SSPCAB, self).__init__(name=name)
        
        self.kernel_dim = kernel_dim
        self.dilation = dilation
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1
        
        self.sspcab_1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu, name=f'{name}_1')
        
        self.sspcab_3 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu, name=f'{name}_3')
        
        self.sspcab_7 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu, name=f'{name}_7')
        
        self.sspcab_9 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu, name=f'{name}_9')
        
    def se_block(self, input_x, in_channels, ratio):
        squeeze = tf.reduce_mean(input_x, axis=[1, 2])
        
        excitation = tf.keras.layers.Dense(use_bias=True, units=in_channels / ratio, name='sspcab_se_1')(squeeze)
        excitation = tf.nn.relu(excitation)
        excitation = tf.keras.layers.Dense(use_bias=True, units=in_channels, name='sspcab_se_2')(excitation)
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, in_channels])
        scale = input_x * excitation
        
        return scale
        
        
    def call(self, inputs, *args, **kwargs):
        sspcab_input = tf.pad(input, tf.constant([[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]]), "REFLECT")
        
        out_1 = self.sspcab_1(sspcab_input[:, :-self.border_input, :-self.border_input, :])
        out_3 = self.sspcab_3(sspcab_input[:, self.border_input:, :-self.border_input, :])
        out_7 = self.sspcab_7(sspcab_input[:, :-self.border_input, self.border_input:, :])
        out_9 = self.sspcab_9(sspcab_input[:, self.border_input:, self.border_input:, :])
        
        sspcab_out = out_1 + out_3 + out_7 + out_9

        se_out = self.se_block(sspcab_out, self.filters, self.reduction_ratio)
        
        cost_sspcab = tf.square(sspcab_input - se_out)
        loss = 0.1 * tf.reduce_mean(cost_sspcab)
        
        self.add_loss(loss)

        return se_out