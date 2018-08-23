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


def n_classifiers(input_layer, regularizer, num_classifiers, class_per_classifier):
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


    
    cls = [tf.layers.dense(inputs=h2,
                           units=class_per_classifier,
                           activation=None,
                           kernel_regularizer=regularizer)
           for _ in range(num_classifiers)]

    print("classifiers : ", cls)
    return cls




def n_classifiers_loss(cls, l_gt):
    assert len(l_gt) = len (cls), "The two len must be equal"
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cl,
        labels=gt)
              for cl,gt in zip(cls, l_gt)]


    print(losses)
    return losses, cls


def five_regressors_loss(input_layer, labels, regularizer):
    return n_classifiers_loss(
        five_classifiers(input_layer, regularizer),
        labels)


def five_classifiers(input_layer, regularizer=None):
    return n_classifiers(input_layer, regularizer, num_classifiers=5, class_per_classifier=3)
