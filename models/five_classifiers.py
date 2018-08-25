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
    # v1 no hidden layer
    # Dense Layer
    input_flat = tf.layers.Flatten()(input_layer)
    print(input_flat)

    cls = [tf.layers.dense(inputs=input_flat,
                           units=class_per_classifier,
                           activation=None,
                           kernel_regularizer=regularizer,
                           name="classif_"+str(num))
           for num in range(num_classifiers)]

    print("classifiers : ", cls)
    return cls



def n_classifiers_loss(cls, labels):
    print("Labels : ", labels)
    l_label = tf.split(labels, 5, 1)
    print("L_label = ", l_label)
    assert len(l_label) == len(cls), "The two len must be equal"
    losses = [tf.losses.sparse_softmax_cross_entropy(
        logits=cl,
        labels=l)
        #weights=tf.tensor([2.0, 1.0, 2.0]))
              for cl, l in zip(cls, l_label)]

    print("Losses : ", losses)
    return losses, cls


def five_classifiers_loss(input_layer, labels, regularizer):
    return n_classifiers_loss(
        five_classifiers(input_layer, regularizer),
        labels)


def five_classifiers(input_layer, regularizer=None):
    return n_classifiers(
        input_layer,
        regularizer,
        num_classifiers=5,
        class_per_classifier=3)
