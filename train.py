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

import argparse
import os
import subprocess
import time
import yaml
import shutil
from datetime import datetime

import tensorflow as tf

from resnet import resnet_v2
from models import five_classifiers


from load_images import create_batch_from_files
slim = tf.contrib.slim


def manage_test_process(queue, f, cpopen):
    if queue:
        if cpopen is not None:
            if cpopen.poll() is None:
                return cpopen  # not finish

        print("OK create new process")

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = ""  # test  on CPU

        argument = queue.pop()
        popen = subprocess.Popen(argument, stdout=f, stderr=f, env=my_env)
        return popen
    else:
        return cpopen


def train_model(hyperparams,
                max_steps,
                data_paths,
                output_path,
                pre_train_path,
                fine_tune,
                test_path,
                debug_flag):
    with tf.device('/cpu:0'):  # data augmentation on CPU to increase perf
        img_batch, label_batch, _ = create_batch_from_files(
            data_paths,
            hyperparams['batch_size'],
            [299, 299],
            3,
            dir_test=test_path,
            data_aug=True,
            training=True,
            threads=True)

    with slim.arg_scope(
            resnet_v2.resnet_arg_scope(
                weight_decay=hyperparams['reg_fact'],
                batch_norm_decay=0.99)):

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
            tf.contrib.layers.l2_regularizer(
                hyperparams['reg_fact'],
                scope=None))
        classifs_loss = tf.add_n(classifs_loss)
        
    train_vars_scope = ['five_classifiers']

    exc_var_load = [scope.strip() for scope in ['five_classifiers']]

    if tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
        if fine_tune:
            assert pre_train_path
            print("** Fine tuning activated, we set up regularization")

            variables_to_regul = []

            for scope in train_vars_scope:
                variables = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES,
                    scope)
                variables_to_regul.extend(variables)

            print(variables_to_regul)
            regulation_losses = tf.add_n(variables_to_regul)

        else:
            print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            regulation_losses = tf.add_n(
                tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
    else:
        regulation_losses = 0.0


    loss = regulation_losses + classifs_loss
    print(loss)

    # tf.summary.scalar("acc_class", accuracy_class)
    tf.summary.scalar("classifs_loss", tf.reduce_mean(classifs_loss))

    l_label = tf.split(label_batch, 5, 1)
    
    l_correct_prediction = [tf.equal(
        tf.cast(
            tf.argmax(
                cl,
                1),
            tf.int32),
        l)
                            for cl, l in zip(l_cl, l_label)]

    
    l_accuracy_class = [tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) for correct_prediction in l_correct_prediction]

    [tf.summary.scalar("acc_class"+n, ac) for n, ac in zip(['O', 'C', 'E', 'A', 'N'], l_accuracy_class)]
    
    tf.summary.scalar("total loss", tf.reduce_mean(loss))

    if not pre_train_path:
        print("No existing model loaded, training from scratch")
        train_operations = slim.learning.create_train_op(
            loss,
            tf.train.AdamOptimizer(hyperparams['learning_rate']))
    else:
        print("Existing model given...")
        exclusions = exc_var_load

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        print("List of variables to load from file")
        for tt in variables_to_restore:
            print(tt.name)

        saver_fine = tf.train.Saver(variables_to_restore)

    if fine_tune:
        assert pre_train_path
        print("** Fine tuning activated, we train only a sublist of existing variables")
        variables_to_train = []

        for scope in train_vars_scope:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope)
            variables_to_train.extend(variables)

        print("Variables to train")
        for tt in variables_to_train:
            print(tt.name)

        train_operations = slim.learning.create_train_op(
            loss,
            tf.train.AdamOptimizer(hyperparams['learning_rate']),
            variables_to_train=variables_to_train)

    else:
        print("** No fine tuning, all variables are trained")
        train_operations = slim.learning.create_train_op(
            loss,
            tf.train.AdamOptimizer(hyperparams['learning_rate']))

    # Create a saver
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()  # for tensorboard

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(output_path, sess.graph)

        if pre_train_path:
            print("Loading existing model from : ", pre_train_path)
            saver_fine.restore(sess, pre_train_path)

        test_process_queue = []
        curr_popen = None

        with open(output_path + "/eval_model_output.log", 'a') as log_file:
            for step in range(max_steps):

                curr_popen = manage_test_process(
                    test_process_queue, log_file, curr_popen)

                start_time = time.time()

                _ = sess.run(train_operations)

                duration = time.time() - start_time

                if step % 100 == 0:
                    num_examples_per_step = hyperparams['batch_size']
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print(format_str % (datetime.now(), step, -1,
                                        examples_per_sec, sec_per_batch))

                    print("wrote sumary op")

                    sum_op, cls_loss, tot_loss, l_a = sess.run(
                        [summary_op, classifs_loss, loss, l_accuracy_class])

                    print("cl_loss = ", cls_loss, "tot_loss = ", tot_loss)
                    print("l_acc = ", l_a)

                    summary_writer.add_summary(sum_op, step)

                if step % 1000 == 0 and step >= 0 and not debug_flag:
                    print("sav checkpoint")
                    checkpoint_path = os.path.join(output_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("done")

                    if test_path:
                        # now launch test
                        print("Enqueue evalution on test set")

                        py_com = ['python3']

                        test_args = py_com + ["./eval.py",
                                              "-tp",
                                              ','.join(test_path),
                                              "-lp",
                                              output_path,
                                              '-dp',
                                              data_paths[0][0],
                                              '-lf',
                                              data_paths[0][1]]
                                              

                        print("*****" , test_args)
                        

                        test_process_queue.insert(0, test_args)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_genera = parser.add_argument_group('General options')
    group_genera.add_argument(
        "-cf",
        "--config_file",
        help="Yaml config file to load",
        required=True)
    args = parser.parse_args()

    with open(args.config_file, 'r') as cfile:
        yml = yaml.load(cfile)

        dropout = yml['dropout']
        reg_fact = yml['reg-factor']
        batch_size = yml['batch-size']
        learning_rate = yml['learning-rate']
        base_path = yml['base-path']
        data_paths = yml['train-paths']
        test_paths = yml['test-paths']
        debug_flag = yml['debug']
        pre_train_path = yml['pre_train_path']

        if dropout == 1:
            dp_str = "No"
            dropout_f = False
        else:
            dp_str = str(dropout)
            dropout_f = True

        if debug_flag:
            output_dir = "DEBUG"
            output_path = os.path.join(base_path, output_dir)
            print("****************** DEBUG MODE ********************")

        else:
            output_dir = "reg_" + str(reg_fact)
            output_dir += "_dp_" + dp_str
            output_dir += "_bs_" + str(batch_size)
            output_dir += "_lr_" + str(learning_rate)
            output_dir += "_ft_" + str(yml['fine-tune'])
            output_path_partial = os.path.join(base_path, output_dir)
            output_path = os.path.join(output_path_partial, yml['append-string'])
        
        try:
            os.makedirs(os.path.join(output_path))
        except OSError:
            print("W pass already exists, pass")


        #copy config file
        shutil.copyfile(args.config_file, output_path + "/config.yaml")

        print("Configuration resume")
        
        print("++++ data path   : ", data_paths)
        print("---- output path : ", output_path)
        print("---- test_paths : ", test_paths)
        
        print("reg fact = ", reg_fact)
        print("dropout = ", dp_str)
        print("Batch size =", batch_size)
        print("learning_rate =", learning_rate)
        print("max steps = ", yml['max_steps'])
        print("Weight to load = ", pre_train_path)
        print("Fine tune? ", yml['fine-tune'])
        
        hyperparams = {}
        hyperparams['learning_rate'] = learning_rate
        hyperparams['reg_fact'] = reg_fact
        hyperparams['batch_size'] = batch_size

        train_model(hyperparams,
                    yml['max_steps'],
                    data_paths,
                    output_path,
                    pre_train_path,
                    yml['fine-tune'],
                    test_paths,
                    debug_flag)
