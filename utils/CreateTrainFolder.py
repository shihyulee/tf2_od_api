import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

import argparse

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def convert_classes(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + " id: " + str(id) + "\n"
        msg = msg + " name: '" + name + "'\n}\n\n"
    return msg[:-1]


def create_label_pbtxt(label_path, output_folder_path):
    with open(label_path, 'r') as f:
        data_list = f.read().splitlines()

    label_pbtxt_path = os.path.join(output_folder_path, "label_map.pbtxt")
    label_map = convert_classes(data_list)
    with open(label_pbtxt_path, "w") as f:
        f.write(label_map)
        f.close()
    print("Create 'label_map.pbtxt' Done")


def Create_TrainFolder(project_dir, project_name, num_classes, num_steps, NN_architecture):
    # Fixed Para
    dataset_name = "dataset"
    label_name = "labels.names"
    output_folder = "tf_training"
    batch_size = 1
    #####

    project_folder = os.path.join(project_dir, project_name)
    dataset_path = os.path.join(project_folder, dataset_name)
    label_path = os.path.join(dataset_path, label_name)

    output_folder_path = os.path.join(project_folder, output_folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Create Label Map (pbtxt)
    create_label_pbtxt(label_path, output_folder_path)

    # Creating TFRecord

    LABEL_MAP_FILE = os.path.join(output_folder_path, "label_map.pbtxt")
    TRAIN_XML_FILE = os.path.join(dataset_path, "Train")
    TRAIN_TF_RECORD_DIR = os.path.join(output_folder_path, "train.record")
    TEST_XML_FILE = os.path.join(dataset_path, "eval")
    TEST_TF_RECORD_DIR = os.path.join(output_folder_path, "test.record")

    tfrecord_train = "python {}/generate_tfrecord.py -x {} -l {} -o {}".format(
        project_dir, TRAIN_XML_FILE, LABEL_MAP_FILE, TRAIN_TF_RECORD_DIR)
    tfrecord_test = "python {}/generate_tfrecord.py -x {} -l {} -o {}".format(
        project_dir, TEST_XML_FILE, LABEL_MAP_FILE, TEST_TF_RECORD_DIR)

    os.system(tfrecord_train)
    os.system(tfrecord_test)

    # Create config

    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile('myproject/sample_config/faster_rcnn_sample.config', "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)

    feature_extractor_dict = {
        "resnet50": "faster_rcnn_resnet50",
        "resnet101": "faster_rcnn_resnet101",
        "inception_resnet_v2": "faster_rcnn_inception_v2"
    }

    feature_extractor_dict = feature_extractor_dict[NN_architecture]

    pipeline.model.faster_rcnn.num_classes = num_classes
    pipeline.model.faster_rcnn.feature_extractor.type = feature_extractor_dict

    pipeline.train_input_reader.label_map_path = LABEL_MAP_FILE
    pipeline.train_input_reader.tf_record_input_reader.input_path[0] = TRAIN_TF_RECORD_DIR

    pipeline.eval_input_reader[0].label_map_path = LABEL_MAP_FILE
    pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = TEST_TF_RECORD_DIR

    pipeline.train_config.num_steps = num_steps
    pipeline.train_config.batch_size = batch_size

    config_output = os.path.join(output_folder_path, "pipeline.config")

    config_text = text_format.MessageToString(pipeline)
    with tf.io.gfile.GFile(config_output, "wb") as f:
        f.write(config_text)

    print("Create 'pipeline.config' Done")


if __name__ == '__main__':
    # project
    project_dir = "myproject"
    project_name = "Project_od_1"

    num_classes = 2
    num_steps = 2000

    #  "resnet50",  "resnet101", "inception_resnet_v2"
    NN_architecture = "resnet101"
    Create_TrainFolder(project_dir, project_name,
                       num_classes, num_steps, NN_architecture)
