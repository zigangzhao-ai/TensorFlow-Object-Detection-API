'''
code by zzg 2020-06-03
'''
# coding:utf-8
import numpy as np
import cv2
import os
import tensorflow as tf
import sys
import os.path

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.protos import string_int_label_map_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
# sess = tf.Session(config=config)

##read video
cap = cv2.VideoCapture("/home/zigangzhao/tensorflow/workspace/training_demo/testvideo/test-Thirdlaneline/video-0411/01234567.h264")

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graphlast.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')
NUM_CLASSES = 2

#Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# sess = tf.Session(graph=detection_graph,config=config)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph,config=config) as sess:
    while True:    
      ret, image_np = cap.read()
      
      # 扩展维度，应为模型 [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # 每个框代表一个物体被侦测到     
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
       #每个分值代表侦测到物体的可信度.  
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # 执行侦测任务 
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # 检测结果的可视化    
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      cv2.imshow('object detection', cv2.resize(image_np,(1200,800)))
      if cv2.waitKey(25) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        break

print("finished!")


 