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

import os
import numpy as np
import tensorflow as tf
from resnet import inception_preprocessing


def read_labeled_image_list_jpeg(l_path_to_read, is_train=False, test_directories=[]):
    filenames_path = []
    print("test path", test_directories)
    for dire, lab_file in l_path_to_read:
        dir_list = os.listdir(dire)

        if is_train:
            for root, _, files in os.walk(dire, topdown=False):
                filenames_path += [os.path.join(root, name_f) for name_f in files
                                   if os.path.basename(root) not in test_directories]
        else:
            for root, _, files in os.walk(dire, topdown=False):
                filenames_path += [os.path.join(root, name_f) for name_f in files
                                   if os.path.basename(root) in test_directories]

        #print(filenames_path)
        dir_list = sorted(
            dir_list,
            key=lambda d: d.lower().replace("_", "+"))

        print(dir_list)
        data = np.loadtxt(lab_file, delimiter=";")
        print("Labels generation done", len(dir_list), len(data))
        assert len(dir_list) == len(data), "The two len must be equals"

        di = {x: data[pos] for pos, x in enumerate(dir_list)}

        #print([os.path.basename(os.path.dirname(f)) for f in filenames_path])
        #now each file should be associated with a label
        labels = [di[os.path.basename(os.path.dirname(f))]
                  for f in filenames_path]

        labels = np.asarray(labels)

        #print(filenames_path[180])
        #print(labels[180])
        print(len(labels), len(filenames_path))

    return filenames_path, labels


def file_operations_eval(tensor_string):
    image_file = tf.read_file(tensor_string)
    image = tf.image.decode_jpeg(image_file, channels=3)

    image = inception_preprocessing.preprocess_for_eval(
        image,
        299,
        299,
        central_fraction=None)
    return image


def file_operations_train(tensor_string):
    image_file = tf.read_file(tensor_string)
    image = tf.image.decode_jpeg(image_file, channels=3)

    image = inception_preprocessing.preprocess_for_train(
        image,
        299,
        299,
        None,
        min_object_covered=0.88,
        aspect_ratio_range=(0.7, 1.4),
        area_range=(0.05, 1.0),
        fast_mode=True)
    return image


def create_batch_from_files(l_data_path,
                            batch_size,
                            l_img_size,
                            chan_num,
                            dir_test,
                            data_aug=False,
                            training=False,
                            threads=False):
    image_list, label_list = read_labeled_image_list_jpeg(
        l_data_path,
        training,
        dir_test)

    label_list = tf.convert_to_tensor(label_list, dtype=tf.float32)

    #if training:
    input_queue = tf.train.slice_input_producer(
        [image_list, label_list],
        shuffle=True)
    #else:
    #    input_queue = tf.train.slice_input_producer(
    #        [image_list, label_list],
    #        shuffle=True, num_epochs=1)

    raw_files = input_queue[0]
    label = input_queue[1]

    if data_aug:
        print("** Data augmentation ON")
        image = file_operations_train(input_queue[0])
    else:
        print("** Data augmentation OFF")
        image = file_operations_eval(input_queue[0])

    if threads:
        read_threads = 10
    else:
        read_threads = 1

    example_list = [(image, label, raw_files) for _ in range(read_threads)]

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    if training:
        image_batch, label_batch, raw_files_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    else:
        image_batch, label_batch, raw_files_batch = tf.train.batch_join(
            example_list,
            batch_size=batch_size,
            capacity=capacity,
            allow_smaller_final_batch=True)

    return image_batch, label_batch, raw_files_batch
