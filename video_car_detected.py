# coding=gbk
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


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto() 
# config.gpu_options.allow_growth = True 
# sess = tf.Session(config=config)

rootdir="video-0411/"

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph20200411.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')
NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# sess = tf.Session(graph=detection_graph,config=config)
	
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
print(image_tensor[0])

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
# count = 0
for parent, dirnames, filenames in os.walk(rootdir):
	print(dirnames)
	for filename in filenames[258:594]:
		# count += 1
		filename1 = os.path.splitext(filename)[0]
		os.mkdir(os.path.join(parent,filename1))
		os.mkdir(os.path.join(parent,filename1,'1'))
		os.mkdir(os.path.join(parent,filename1,'2'))
		s1=os.path.join(parent,filename1,'1')
		print(s1)
		s2=os.path.join(parent,filename1,'2')
		print(s2)
		cap = cv2.VideoCapture(os.path.join(parent,filename))
		i=0
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
				break
			i+=1
			# s=s1+"/"+filename1+'-'+str(i)+'.jpg'
			# s = s1+str(i)+'.jpg'
			# cv2.imwrite(s,frame)
			src_frame = frame.copy()
			image_expanded = np.expand_dims(frame, axis=0)
			(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_expanded})
			vis_util.visualize_boxes_and_labels_on_image_array(
			frame,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			category_index,
			use_normalized_coordinates=True,
			line_thickness=8,
			min_score_thresh=0.80)
			####check
			check_result = 0
			print(np.squeeze(scores).shape[0])
			scores_np = np.squeeze(scores).reshape(np.squeeze(scores).shape[0],-1)
			for box_idx in range(np.squeeze(scores).shape[0]):
			    if (scores_np[box_idx] > 0.8):
				    check_result = 1
				    break
			if(check_result):
			    s = s1 + "/" + filename1 + '-' + str(i) + '.jpg' 
			    cv2.imwrite(s,src_frame)	

			    s = s2 + "/" + filename1 + '-' + str(i) + '.jpg'			    
			    cv2.imwrite(s,frame)

			# cv2.imshow('frame',frame)
			# s=s2+"/"+filename1+'-'+str(i)+'.jpg'
			# cv2.imwrite(s,frame)
		cap.release()
print("finished!")

