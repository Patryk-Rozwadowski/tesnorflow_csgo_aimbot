import cv2
import numpy as np
import os
import tensorflow as tf
import time
from mss import mss

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from settings import object_detection_paths

width = 1200
height = 900

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sct = mss()

dot_thickness = 2
settings = {
    'title': 'Game vision',
    'mss_mon': {"top": 40, "left": 0, "width": width, "height": height},
}

paths = object_detection_paths.paths
NUM_CLASSES = 4

label_map = label_map_util.load_labelmap(paths['labels'])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()


def getTargetAndShoot(arr):
    array_ct_head = []
    array_ct = []

    array_tt_head = []
    array_tt = []

    for i, b in enumerate(arr):
        if classes[0][i] == 2:  # ct_head

            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                array_ct_head.append([mid_x, mid_y])
                cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), dot_thickness, (0, 0, 255), -1)

        if classes[0][i] == 1:  # ct
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                array_ct.append([mid_x, mid_y])
                cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), dot_thickness, (50, 150, 255), -1)

        if classes[0][i] == 4:  # tt_head
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                array_tt_head.append([mid_x, mid_y])
                cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), dot_thickness, (0, 0, 255), -1)

        if classes[0][i] == 3:  # t
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                array_tt.append([mid_x, mid_y])
                cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), dot_thickness, (50, 150, 255), -1)


with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(paths['frozen_graph'], 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        start_time = time.time()
        fps_log_refresh = 2
        fps = 0

        while True:
            image_np = np.array(sct.grab(settings['mss_mon']))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            getTargetAndShoot(boxes[0])
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)

            # Show image with detection
            cv2.imshow(settings['title'], cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

            fps += 1
            TIME = time.time() - start_time
            if TIME >= fps_log_refresh:
                print("FPS: ", fps / TIME)
                fps = 0
                start_time = time.time()

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
