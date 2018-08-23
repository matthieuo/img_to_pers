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
from resnet import resnet_v2
from models import five_regressors
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

    with tf.variable_scope('five_regressors'):
        regressor_loss, rg_no_sum, _, ro = five_regressors.five_regressors_loss(
            out_resnet,
            label_batch,
            None)
  
    tf.summary.scalar("MSE", regressor_loss)
    tf.summary.scalar("RMSE", tf.sqrt(regressor_loss))

    tf.summary.tensor_summary("MSE", rg_no_sum)
    tf.summary.histogram("MSE", rg_no_sum)

    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()

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

        lb, loss_v, loss_v_ns, sum_op, r_out = sess.run(
            [label_batch, regressor_loss, rg_no_sum, summary_op, ro])

        print("Mean result", np.mean(r_out, axis=0))
        print("Objective ", lb)
        
        print("nosum ", np.mean(loss_v_ns, axis=0))
        print("nosum sq ", np.sqrt(np.mean(loss_v_ns, axis=0)))
        print("MSE : ", loss_v)
        print("RMSE : ", np.sqrt(loss_v))


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
