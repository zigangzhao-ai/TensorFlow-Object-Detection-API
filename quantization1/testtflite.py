'''
code by zzg 2020-04-30
'''
import tensorflow as tf
import numpy as np

InputSize = 300

def test_tflite(input_test_tflite_file):
    interpreter = tf.lite.Interpreter(model_path = input_test_tflite_file)
    tensor_details = interpreter.get_tensor_details()
    for i in range(0,len(tensor_details)):
        # print("tensor:", i, tensor_details[i])
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print("=======================================")
    print("input :", str(input_details))
    output_details = interpreter.get_output_details()
    print("ouput :", str(output_details))
    print("=======================================")
    new_img = np.random.uniform(0,1,(1,InputSize,InputSize,3))
    # image_np_expanded = np.expand_dims(new_img, axis=0)
    new_img = new_img.astype('uint8')# 类型也要满足要求

    interpreter.set_tensor(input_details[0]['index'],new_img)
    # 注意注意，我要调用模型了
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("test_tflite finish!")


intput_tflite_file = "uint8.tflite"
test_tflite(intput_tflite_file)