import os
import sys
import random
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
import skimage.io

import coco
import utils
import model as modellib
import visualize

SRC_IMAGE_PATH = sys.argv[1]
DST_IMAGE_PATH = sys.argv[2]

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#file_names = "000000503066.jpg"
#file_names = "8053677163_d4c8f416be_z.jpg"
#image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names))

image = skimage.io.imread(SRC_IMAGE_PATH)

# Run detection
results = model.detect([image], verbose=1)
r = results[0]

visualize.display_instances_x(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], save_path=os.path.join(ROOT_DIR, "./example-pre.png"))

from instance_context_model import *

instance_context_graph = tf.Graph()
with instance_context_graph.as_default():
    learning_rate = 1e-3
    init, inputs_, labels_, \
    final_output, loss, \
    train_step, saver = build_model_graph(learning_rate)

with tf.Session(graph=instance_context_graph) as session:
    model_restore(session, saver, model_base_path = './models.1/')
    ins_catIds = get_coco_catIds(set(r['class_ids']))
    ret = validate_one(session, ins_catIds, inputs_, final_output)
    odd_object_id = list(r['class_ids']).index(cocoCatIds.index(ret[-1]) + 1)
    zeros = np.zeros(shape=(4,), dtype=np.int32)
    r['rois'][0:odd_object_id] = zeros
    r['rois'][odd_object_id+1:] = zeros

visualize.display_instances_x(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], save_path=DST_IMAGE_PATH)
print ("OK!")
