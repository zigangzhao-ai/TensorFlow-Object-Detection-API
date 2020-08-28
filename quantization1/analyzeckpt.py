<<<<<<< HEAD
"""
code by zzg
"""
from tensorflow.python import pywrap_tensorflow 
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 


checkpoint_path = "/home/zigangzhao/tensorflow-car/workspace/training_demo/training0504/model.ckpt-18200"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
var_to_shape_map = reader.get_variable_to_shape_map() 
for key in var_to_shape_map: 
    print("tensor_name: ", key)
=======
"""
code by zzg
"""
from tensorflow.python import pywrap_tensorflow 
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 


checkpoint_path = "/home/zigangzhao/tensorflow-car/workspace/training_demo/training0504/model.ckpt-18200"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) 
var_to_shape_map = reader.get_variable_to_shape_map() 
for key in var_to_shape_map: 
    print("tensor_name: ", key)
>>>>>>> 23df4cf87831b2e469416fd8cee338d2afd957a3
