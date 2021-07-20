"""
Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Neural net Models"""


# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
# stderr = sys.stderr
# sys.stderr = open(os.devnull, "w")
# sys.stderr = stderr




# from ae.loss import get_loss
from keras.utils import plot_model
from keras.regularizers import l1
from keras import optimizers
from keras.models import Model
import math
import numpy as np
import os
import keras
import sys
from keras.layers import Input, Dense, Dropout, Conv1D,  UpSampling1D,  Cropping1D,   ZeroPadding1D,   Flatten,   Reshape,    BatchNormalization
from keras import backend as K

from tensorflow.losses import huber_loss
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config));



# sys.stderr = stderr

eps = np.float(K.epsilon())


def scaled_mean_squared_error(scale: float = 1.0, with_numpy: bool = False):
    """Scaled mean squared error

    Scaling is applied to the absolute error before squaring the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- MSE of the scaled error
    """

    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true) * scale), axis=-1)

    def mean_squared_error_numpy(y_true, y_pred):
        return np.mean(np.square((y_pred - y_true) * scale), axis=-1)

    if with_numpy:
        return mean_squared_error_numpy

    return mean_squared_error


def scaled_mean_absolute_error(scale: float = 1.0):
    """Scaled mean absolute error

    Scaling is applied to the absolute error before taking the absolute the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- MAE of the scaled error
    """

    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs((y_pred - y_true) * scale), axis=-1)

    return mean_absolute_error


def scaled_logcosh(scale: float = 1.0):
    """Scale logcosh loss

    Scaling is applied to the absolute error before logcoshing the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- Logcosh of the scaled error
    """

    def _logcosh(x):
        return x + K.softplus(-2.0 * x) - K.log(2.0)

    def logcosh(y_true, y_pred):
        return K.mean(_logcosh((y_pred - y_true) * scale), axis=-1)

    return logcosh


def scaled_huber(scale: float = 1.0, delta: float = 1.0):
    """Scaled Huber loss

    Scaling is applied to the absolute error before hubering the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})
        delta {float} -- Huber's delta parameter (default: {1})

    Returns:
        {tensor} -- Huber loss of the scaled error
    """

    def huber(y_true, y_pred):
        # return Huber(y_true * scale, y_pred * scale, delta=delta)
        return huber_loss(y_true * scale, y_pred * scale, delta=delta)

    return huber


def binary_crossentropy_numpy(y_true, y_pred):
    output = np.clip(y_pred, eps, 1 - eps)

    return np.mean(
        -(y_true * np.log(output) + (1 - y_true) * np.log(1 - output)), axis=-1
    )




def get_loss(loss: str):
    loss_parts = loss.split("-")

    if loss.startswith("smse") and len(loss_parts) > 1:
        loss = scaled_mean_squared_error(float(loss_parts[1]))

    elif loss.startswith("smae") and len(loss_parts) > 1:
        loss = scaled_mean_absolute_error(float(loss_parts[1]))

    elif loss.startswith("shuber") and len(loss_parts) > 2:
        loss = scaled_huber(float(loss_parts[1]), float(loss_parts[2]))

    elif loss.startswith("slogcosh") and len(loss_parts) > 1:
        loss = scaled_logcosh(float(loss_parts[1]))

    elif loss.startswith("bce"):
        loss = "binary_crossentropy"

    return loss

def create_model(windowSize: int, n_features :int):
    input_sig = Input(batch_shape=(None,windowSize,n_features))
    x = Conv1D(128,3, activation='relu', strides = 2, padding='same')(input_sig)
    x1 = Conv1D(192,5, activation='relu', strides = 2, padding='same')(x)
    x2 = Conv1D(288,7, activation='relu', strides = 2, padding='same')(x1)
    x3 = Conv1D(432,9, activation='relu', strides = 2, padding='same')(x2)
    flat = Flatten()(x3)
    fc4 = Dense(1024)(flat)
    fc5 = Dense(256)(fc4)
    embed = Dense(10)(fc5)
    fc6 = Dense(256)(embed)
    fc7 = Dense(1024)(fc6)
    blowup = Dense(3456)(fc7)
    unflatten = Reshape((8, 432))(blowup) # , input_shape=(None,3456)
    upsample0 = UpSampling1D(size=2)(unflatten)
    cropping0 = Cropping1D(cropping=(1,0))(upsample0)
    deconv0 = Conv1D(288, 9, activation='relu', padding='same')(cropping0)
    upsample1 = UpSampling1D(size=2)(deconv0)
    deconv1 = Conv1D(192, 7, activation='relu', padding='same')(upsample1)
    upsample2 = UpSampling1D(size=2)(deconv1)
    deconv2 = Conv1D(128, 5, activation='relu', padding='same')(upsample2)
    upsample3 = UpSampling1D(size=2)(deconv2)
    output = Conv1D(3, 3, padding='same')(upsample3)
    model= Model(input_sig, output)
    model.summary()
    return model

# def create_model(
#     input_dim: int,
#     optimizer: str = "adadelta",
#     loss: str = "smse-10",
#     conv_filters: list = [120],
#     conv_kernels: list = [9],
#     dense_units: list = [256, 64, 16],
#     embedding: int = 10,
#     dropouts: list = [0.0, 0.0, 0.0],
#     batch_norm: list = [False, False, False],
#     batch_norm_input: bool = False,
#     metrics: list = [],
#     reg_lambda: float = 0.0,
#     learning_rate: float = 1.0,
#     learning_rate_decay: float = 0.001,
#     summary: bool = False,
#     plot: bool = False,
#     n_features: int = 1,
# ):
#     inputs = Input(shape=(input_dim, 2), name="input")

#     num_cfilter = len(conv_filters)
#     num_dunits = len(dense_units)
#     num_batch_norm = len(batch_norm)

#     if len(batch_norm) < num_cfilter + num_dunits:
#         batch_norm += [False] * (num_cfilter + num_dunits - num_batch_norm)

#     encoded = inputs
#     if batch_norm_input:
#         encoded = BatchNormalization(axis=1)(encoded)

#     input_sizes = []

#     for i, f in enumerate(conv_filters):
#         encoded = Conv1D(
#             f,
#             conv_kernels[i],
#             strides=2,
#             activation="relu",
#             padding="same",
#             name="conv{}".format(i),
#         )(encoded)
#         input_sizes.append(int(encoded.shape[n_features]))
#         if batch_norm[i]:
#             encoded = BatchNormalization(axis=1)(encoded)
#         if dropouts[i] > 0:
#             encoded = Dropout(dropouts[i], name="drop{}".format(i))(encoded)

#     encoded = Flatten(name="flatten")(encoded)

#     for i, u in enumerate(dense_units):
#         k = num_cfilter + i
#         encoded = Dense(u, activation="relu", name="fc{}".format(k))(encoded)
#         if batch_norm[i]:
#             encoded = BatchNormalization(axis=1)(encoded)
#         if dropouts[i] > 0:
#             encoded = Dropout(dropouts[i], name="drop{}".format(k))(encoded)

#     # The bottleneck that will hold the latent representation
#     encoded = Dense(
#         embedding, activation="relu", name="embed", kernel_regularizer=l1(reg_lambda)
#     )(encoded)

#     decoded = encoded
#     for i, u in enumerate(reversed(dense_units)):
#         k = num_cfilter + num_dunits + i
#         decoded = Dense(u, activation="relu", name="fc{}".format(k))(decoded)
#         if dropouts[i] > 0:
#             decoded = Dropout(dropouts[i], name="dropout{}".format(k))(decoded)

#     decoded = Dense(
#         input_sizes[-1] * conv_filters[-1], activation="relu", name="blowup"
#     )(decoded)
#     decoded = Reshape(
#         (input_sizes[-1], conv_filters[-1]), name="unflatten")(decoded)

#     for i, f in enumerate(reversed(conv_filters[:-1])):
#         k = num_cfilter + (num_dunits * 2) + i
#         j = num_cfilter - i - 2
#         decoded = UpSampling1D(2, name="upsample{}".format(i))(decoded)
#         diff = int(decoded.shape[1]) - input_sizes[-(i + 2)]
#         if diff > 0:
#             left_cropping = math.floor(diff / 2)
#             right_cropping = math.ceil(diff / 2)
#             decoded = Cropping1D(
#                 cropping=(left_cropping,
#                           right_cropping), name="cropping{}".format(i)
#             )(decoded)
#         elif diff < 0:
#             left_padding = math.floor(math.abs(diff) / 2)
#             right_padding = math.ceil(math.abs(diff) / 2)
#             decoded = ZeroPadding1D(
#                 cropping=(left_padding,
#                           right_padding), name="padding{}".format(i)
#             )(decoded)
#         decoded = Conv1D(
#             f,
#             conv_kernels[:-1][j],
#             activation="relu",
#             padding="same",
#             name="deconv{}".format(i),
#         )(decoded)
#         if dropouts[i] > 0:
#             decoded = Dropout(dropouts[i], name="drop{}".format(k))(decoded)

#     decoded = UpSampling1D(2, name="upsample{}".format(
#         len(conv_filters) - 1))(decoded)
#     decoded = Conv1D(
#         n_features, conv_kernels[0], activation="sigmoid", padding="same", name="output"
#     )(decoded)

#     autoencoder = Model(inputs, decoded)

#     if optimizer == "sgd":
#         opt = optimizers.SGD(lr=learning_rate, decay=learning_rate_decay)

#     elif optimizer == "nesterov":
#         opt = optimizers.SGD(
#             lr=learning_rate, decay=learning_rate_decay, nesterov=True)

#     elif optimizer == "rmsprop":
#         opt = optimizers.RMSprop(lr=learning_rate, decay=learning_rate_decay)

#     elif optimizer == "adadelta":
#         opt = optimizers.Adadelta(lr=learning_rate, decay=learning_rate_decay)

#     elif optimizer == "adagrad":
#         opt = optimizers.Adagrad(lr=learning_rate, decay=learning_rate_decay)

#     elif optimizer == "adam" or optimizer[:5] == "adam-":
#         try:
#             beta_1 = float(optimizer.split("-")[1])
#         except IndexError:
#             beta_1 = 0.9

#         try:
#             beta_2 = float(optimizer.split("-")[2])
#         except IndexError:
#             beta_2 = 0.999

#         opt = optimizers.Adam(
#             lr=learning_rate, decay=learning_rate_decay, beta_1=beta_1, beta_2=beta_2
#         )

#     elif optimizer == "amsgrad":
#         opt = optimizers.Adam(
#             lr=learning_rate, decay=learning_rate_decay, amsgrad=True)

#     elif optimizer == "adamax":
#         opt = optimizers.Adamax(lr=learning_rate, decay=learning_rate_decay)

#     elif optimizer == "nadam":
#         opt = optimizers.Adamax(lr=learning_rate)

#     else:
#         print("Unknown optimizer: {}. Using Adam.".format(optimizer))
#         opt = optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)

#     loss = get_loss(loss)
#     autoencoder.compile(optimizer=opt, loss=loss, metrics=metrics)

#     encoder = Model(inputs, encoded)

#     encoded_input = Input(shape=(embedding,), name="encoded_input")
#     decoded_input = encoded_input
#     num_dropout_units = sum([1 if x > 0 else 0 for x in dropouts])
#     k = (
#         num_dunits
#         + num_cfilter
#         + num_dropout_units
#         + sum(batch_norm)
#         + int(batch_norm_input)
#         + 3
#     )
#     for i in range(k, len(autoencoder.layers)):
#         decoded_input = autoencoder.layers[i](decoded_input)
#     decoder = Model(encoded_input, decoded_input)

#     if summary:
#         print(autoencoder.summary())
#         print(encoder.summary())
#         print(decoder.summary())

#     if plot:
#         plot_model(
#             autoencoder, to_file="cnn3_ae.png", show_shapes=True, show_layer_names=True
#         )
#         plot_model(
#             encoder, to_file="cnn3_de.png", show_shapes=True, show_layer_names=True
#         )
#         plot_model(
#             encoder, to_file="cnn3_en.png", show_shapes=True, show_layer_names=True
#         )

#     return (encoder, decoder, autoencoder)


if __name__ == "__main__":
    T = 20
    N = 10
    F = 1
    model = create_model(windowSize=T, n_features = F)
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    input_data = np.zeros([N, T,F])
    model.fit(input_data, input_data)