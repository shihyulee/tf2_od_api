from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf
from object_detection import model_lib
import os

def model_train_val(MODEL_DIR, config_output):
    config = tf.estimator.RunConfig(model_dir=MODEL_DIR)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      pipeline_config_path=config_output)
    estimator = train_and_eval_dict['estimator']
    
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
	
if __name__ == '__main__':
	project_dir = "myproject"
	project_name = "Project_od_1"
	dataset_name = "dataset"
	label_name = "labels.names"
	output_folder = "tf_training"
	project_folder = os.path.join(project_dir, project_name)
	output_folder_path =  os.path.join(project_folder, output_folder)

	MODEL_DIR = os.path.join(output_folder_path, "train")
	config_output = os.path.join(output_folder_path, "pipeline.config")

	model_train_val(MODEL_DIR, config_output)