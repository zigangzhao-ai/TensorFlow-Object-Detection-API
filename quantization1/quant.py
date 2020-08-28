'''
code by zzg 2020-04-27
'''


import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 

graph_def_file = "frozen_inference_graph0428.pb"


input_names = ["FeatureExtractor/MobilenetV2/MobilenetV2/input"]
# input_names = ["FeatureExtractor/resnet_v1_50/resnet_v1_50/Pad/paddings"] #/Pad/paddings
# print("=========================")
# print(input_names)
# print(input_names[0])
output_names = ["concat", "concat_1"]
# input_tensor = {input_names[0]:[4,2]}
input_tensor = {input_names[0]:[1,300,300,3]}

#pb-->tflite
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
tflite_fp32_model = converter.convert()
open("fp32.tflite", "wb").write(tflite_fp32_model)

#fp16 quant  --tf1.15可以转换
# converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file1, input_names, output_names, input_tensor)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
# tflite_fp16_model = converter.convert()
# open("p16.tflite", "wb").write(tflitef_p16_model)

#int8 quant
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True

converter.optimizations = [tf.lite.Optimize.DEFAULT]  #都可以，混合量化，仅量化权重
# converter.post_training_quantize=True  
tflite_int8_model = converter.convert()
open("int8.tflite", "wb").write(tflite_int8_model)


#unit8 quant
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True

converter.inference_type = tf.uint8    #tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (0, 255)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)
#converter.inference_type = tf.uint8    #tf.lite.constants.QUANTIZED_UINT8

tflite_uint8_model = converter.convert()
open("uint8.tflite", "wb").write(tflite_uint8_model)