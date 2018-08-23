#    img_to_pers
#    Copyright (C) 2018  Matthieu Ospici
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf


def n_regressors_back(input_layer, regularizer, num_regressors):
    # Dense Layer
    input_flat = tf.layers.Flatten()(input_layer)
    print(input_flat)
    regs = tf.layers.dense(inputs=input_flat,
                           units=num_regressors,
                           activation=None,
                           kernel_regularizer=regularizer)

    print("regressors : ", regs)
    return regs


def n_regressors(input_layer, regularizer, num_regressors):
    # v1: one hidden layer, 500 units out
    # v2: two hidden layer, 900 and 300 units out
    # Dense Layer
    input_flat = tf.layers.Flatten()(input_layer)
    print(input_flat)

    h1 = tf.layers.dense(inputs=input_flat,
                         units=900,
                         activation=tf.nn.relu,
                         kernel_regularizer=regularizer)

    h2 = tf.layers.dense(inputs=h1,
                         units=300,
                         activation=tf.nn.relu,
                         kernel_regularizer=regularizer)

    regs = tf.layers.dense(inputs=h2,
                           units=num_regressors,
                           activation=None,
                           kernel_regularizer=regularizer)

    print("regressors : ", regs)
    return regs




def n_regressors_loss(regs, gt):
    loss = tf.losses.mean_squared_error(gt, regs)
    loss_no_sum = tf.losses.mean_squared_error(
        gt,
        regs, reduction=tf.losses.Reduction.NONE)
    log_loss = tf.losses.log_loss(gt, regs)

    print(loss, loss_no_sum, log_loss)
    return loss, loss_no_sum, log_loss, regs


def five_regressors_loss(input_layer, labels, regularizer):
    return n_regressors_loss(
        n_regressors(input_layer, regularizer, 5),
        labels)


def five_regressors(input_layer, regularizer=None):
    return n_regressors(input_layer, regularizer, 5)
