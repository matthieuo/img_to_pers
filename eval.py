#    img_to_pers
#    Copyright (C) 2018 Matthieu Ospici
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

import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from resnet import resnet_v2
from models import five_classifiers
from load_images import create_batch_from_files
slim = tf.contrib.slim


def test_model(data_path,
               test_path,
               log_path):
    #print(test_path)
    with tf.device('/cpu:0'):  # data augmentation on CPU to increase perf
        img_batch, label_batch, _ = create_batch_from_files(
            data_path,
            500,
            [299, 299],
            3,
            dir_test=test_path,
            data_aug=False,
            training=False,
            threads=True)

    with slim.arg_scope(
            resnet_v2.resnet_arg_scope(
                batch_norm_decay=0.9)):

        out_resnet, _ = resnet_v2.resnet_v2_50(
            img_batch,
            None,
            is_training=True,
            global_pool=True,
            spatial_squeeze=True)

    with tf.variable_scope('five_classifiers'):
        classifs_loss, l_cl = five_classifiers.five_classifiers_loss(
            out_resnet,
            label_batch,
            None)

    l_label = tf.split(label_batch, 5, 1)
    
    l_correct_prediction = [tf.equal(
        tf.cast(
            tf.argmax(
                cl,
                1),
            tf.int32),
        tf.squeeze(l))
                            for cl, l in zip(l_cl, l_label)]
    
    l_accuracy_class = [tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) for correct_prediction in l_correct_prediction]

    [tf.summary.scalar("test_acu/" + n, ac) for n, ac in zip(['O', 'C', 'E', 'A', 'N'], l_accuracy_class)]



    l_ap = [tf.metrics.auc(tf.squeeze(l), tf.argmax(pred, 1)) for pred, l in zip(l_cl, l_label)]

       
    [tf.summary.scalar("test_AP/" + n, ap[0]) for n, ap in zip(['O', 'C', 'E', 'A', 'N'], l_ap)]

    

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    summary_op = tf.summary.merge_all()

    
    with tf.Session() as sess:
        sess.run(init_op)

        # create queues to load images
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        last_chk = tf.train.latest_checkpoint(log_path)

        summary_writer = tf.summary.FileWriter(log_path+"_eval", sess.graph)
        
        chk_step = last_chk.split("-")[-1]
        print(chk_step)
        saver.restore(sess, last_chk)

        lb, sum_op, llb, llcl = sess.run(
            [label_batch, summary_op, l_label, l_cl])

        for yt, yp in zip(llb, llcl):
            #print(yt.shape, np.argmax(yp, 1).shape)
            print(classification_report(yt, np.argmax(yp, 1)))

        summary_writer.add_summary(sum_op, chk_step)
        
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')

    group_genera.add_argument(
        "-tp",
        "--test_paths",
        help="Name of the paths for test",
        required=True)

    group_genera.add_argument(
        "-dp",
        "--data_path",
        help="Path to the dataset",
        required=True)

    group_genera.add_argument(
        "-lf",
        "--label_file",
        help="Path to the label file",
        required=True)
        
    group_genera.add_argument(
        "-lp",
        "--log_path",
        help="Log directory path",
        required=True)

    args = parser.parse_args()

    print("++++ data path   : ", args.data_path)
    print("++++ test directory  : ", args.test_paths)
    print("++++ log path : ", args.log_path)

    data = [(args.data_path, args.label_file)]
    test_model(data,
               [item for item in args.test_paths.split(',')],
               args.log_path)
