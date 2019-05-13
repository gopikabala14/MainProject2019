# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import sqlite3
import time
import os
sum = 0
flag =0
truck_count=0
seven_count = 0
six_count=0
five_count=0
four_count=0
three_count=0
two_count=0
bus_count=0
car_count=0
seven_sum=0
six_sum=0
five_sum=0
four_sum=0
three_sum=0
two_sum=0
bus_sum =0
car_sum =0

from timeit import default_timer as timer
# from vehicle_class import read
import numpy as np
from keras import backend as K
from keras import backend as k
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import json

dict2 = {}
a=[]
import os
import sys
import itertools
import math
import logging
import json
import re
import random
import skimage
from collections import OrderedDict
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import tensorflow as tf
# Root directory of the projectjup
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import custom
eight_count=0
seven_count=0
six_count=0
five_count=0
four_count=0
three_count=0
two_count=0
bus_count=0
car_count=0

truck_sum=0
eight_sum=0
seven_sum=0
six_sum=0
five_sum=0
four_sum=0
three_sum=0
two_sum=0
bus_sum=0
car_sum=0

# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#
# custom_WEIGHTS_PATH = "mask_rcnn_wheel_0100.h5"  # TODO: update this path
#
#
#
# config = custom.CustomConfig()
# custom_DIR = os.path.join(ROOT_DIR, "customImages")

class YOLO(object):
    _defaults = {
        # "model_path": 'model_data/yolo.h5',                   # 0 original yolo model
        # "model_path": 'model_data/derived_model.h5',         # 1 to test the derived model for coco-dataset
        "model_path": 'model_data/pedestrian_detection_model.h5',           # 2-1) to test the raccoon_dataset_derived_model
        "anchors_path": 'model_data/yolo_anchors.txt',
        # "classes_path": 'model_data/coco_classes.txt',
        "classes_path": 'model_data/trafficlight_classes.txt',         # 2-2) to test the raccoon_dataset_derived_model
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            print('output_shape = %d' %(self.yolo_model.layers[-1].output_shape[-1]))
            print('num_anchors = %d' % num_anchors)
            print('len = %d' %(len(self.yolo_model.output) * (num_classes + 5)))
            print('len_output = %d' %(len(self.yolo_model.output)))
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors/len(self.yolo_model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,k):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5))
            left = max(0, np.floor(left + 0.5))
            bottom = min(image.size[1], np.floor(bottom + 0.5))
            right = min(image.size[0], np.floor(right + 0.5))
            print(label, (left, top), (right, bottom))

            #a=[1,2,3,4,5,6,7,8,9,10]

            dict1={"class":label, i:[left, top, right, bottom]}
            #print(dict1)
            dict2.update(dict1)
            # print(dict2)
            dict1 = {}




            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            conn = sqlite3.connect('toll_data.db')
            c = conn.cursor()

            print("classssssssssss",out_classes)
            ##################################################### MASK RCNN#########################################################
            for i in out_classes:
                if(i==7):
                    # Directory to save logs and trained model
                    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

                    custom_WEIGHTS_PATH = "mask_rcnn_wheel_0100.h5"  # TODO: update this path

                    config = custom.CustomConfig()
                    custom_DIR = os.path.join(ROOT_DIR, "customImages")

                    # ---------------------------------------------------------------------------

                    # Override the training configurations with a few
                    # changes for inferencing.
                    class InferenceConfig(config.__class__):
                        # Run detection on one image at a time
                        GPU_COUNT = 1
                        IMAGES_PER_GPU = 1

                    config = InferenceConfig()
                    config.display()

                    # Device to load the neural network on.
                    # Useful if you're training a model on the same
                    # machine, in which case use CPU and leave the
                    # GPU for training.
                    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

                    # Inspect the model in training or inference modes
                    # values: 'inference' or 'training'
                    # TODO: code for 'training' te

                    def get_ax(rows=1, cols=1, size=16):
                        """Return a Matplotlib Axes array to be used in
                        all visualizations in the notebook. Provide a
                        central point to control graph sizes.

                        Adjust the size attribute to control how big to render images
                        """
                        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
                        return ax

                    # Load validation dataset
                    dataset = custom.CustomDataset()
                    dataset.load_custom(custom_DIR, "val")

                    # Must call before using the dataset
                    dataset.prepare()

                    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

                    # Create model in inference mode
                    with tf.device(DEVICE):
                        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                                  config=config)

                    # load the last model you trained
                    # weights_path = model.find_last()[1]

                    # Load weights
                    print("Loading weights ", custom_WEIGHTS_PATH)
                    model.load_weights(custom_WEIGHTS_PATH, by_name=True)

                    # Display results
                    import skimage
                    # i=read()
                    imag =skimage.io.imread(k)
                    results = model.detect([imag], verbose=1)
                    ax = get_ax(1)
                    r = results[0]
                    visualize.display_instances(imag, r['rois'], r['masks'], r['class_ids'],
                                                dataset.class_names, r['scores'], ax=ax,
                                                title="Predictions")
                    visualize.display_images(imag)
                    print(r['scores'])
                    axle = (len(r['scores']))
                    print(len(r['scores']))
                    if axle >= 8:
                        c = conn.cursor()
                        #seven_count = seven_count + 1
                        global eight_count
                        eight_count = eight_count + 1
                        print(eight_count)
                        print("8 Axle or 8 Axle Above Truck... Pay 12 Riyal")
                        now = datetime.datetime.now()
                        print(now)
                        global eight_sum
                        eight_sum = eight_sum + 12
                        print("eight_sum", eight_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")
                        car = '--'
                        bus = '--'
                        truck = '8 axle or above'
                        amount = 20

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()
                    elif axle == 7:
                        c = conn.cursor()
                        #seven_count = seven_count + 1
                        global seven_count
                        seven_count = seven_count + 1
                        print(seven_count)
                        print("7 Axle Truck... Pay 10 Riyal")
                        now = datetime.datetime.now()
                        print(now)
                        global seven_sum
                        seven_sum = seven_sum + 10
                        print("seven_sum", seven_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")
                        car = '--'
                        bus = '--'
                        truck = '7 axle'
                        amount = 20

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()

                    elif axle == 6:
                        c = conn.cursor()
                        global  six_count
                        six_count = six_count + 1
                        print(six_count)
                        print("6 Axle Truck... Pay 8 Riyal")
                        now = datetime.datetime.now()
                        print(now)
                        global six_sum
                        six_sum = six_sum + 8
                        print("six_sum", six_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")
                        car = '--'
                        bus = '--'
                        truck = '6 axle'
                        amount = 20

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()
                    elif axle == 5:
                        c = conn.cursor()
                        global five_count
                        five_count = five_count + 1
                        print(five_count)
                        print("5 Axle Truck... Pay 6 Riyal")
                        now = datetime.datetime.now()
                        print(now)
                        global five_sum
                        five_sum = five_sum + 6
                        print("five_sum", five_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")
                        car = '--'
                        bus = '--'
                        truck = '5 axle'
                        amount = 20

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()
                    elif axle == 4:
                        c = conn.cursor()
                        global four_count
                        four_count = four_count + 1
                        print(four_count)
                        print("4 Axle Truck... Pay 4 Riyal")
                        now = datetime.datetime.now()
                        print(now)
                        global four_sum
                        four_sum = four_sum + 4
                        print("four_sum", four_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")
                        car = '--'
                        bus = '--'
                        truck = '4 axle'
                        amount = 20

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()
                    elif axle == 3:
                        c = conn.cursor()

                        #global three_count
                        #three_count = three_count + 1
                        #print(three_count)

                        now = datetime.datetime.now()
                        #print(now)
                        #global three_sum
                        #three_sum = three_sum + 2
                        #print("three_sum", three_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")

                        car = '--'
                        bus = '--'
                        truck = '3 axle'
                        amount = 15


                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))
                        conn.commit()


                    else:
                        c = conn.cursor()

                        now = datetime.datetime.now()
                        #global two_sum
                        #two_sum = two_sum + 1

                        #print("two_sum", two_sum)
                        ctime = now.strftime("%I:%M:%S %p")
                        today = now.strftime("%Y-%m-%d")

                        car = '--'
                        bus = '--'
                        truck = '2 axle'
                        amount = 10

                        c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                                  (today, ctime, car, bus, truck, amount))

                        conn.commit()


                elif i == 5:
                    c = conn.cursor()
                    global flag
                    flag = 1
                    now = datetime.datetime.now()

                    #global bus_sum
                    #bus_sum = bus_sum + 3
                    #print(bus_sum)
                    ctime= now.strftime("%I:%M:%S %p")
                    today = now.strftime("%Y-%m-%d")
                    car = '--'
                    bus = 'bus'
                    truck = '--'
                    amount = 20

                    c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                              (today, ctime, car, bus, truck, amount))

                    conn.commit()
                elif i == 2:
                    c = conn.cursor()
                    global flag
                    flag = 2
                    now = datetime.datetime.now()

                    #global car_sum
                    #car_sum = car_sum + 2
                    #print(car_sum)
                    ctime = now.strftime("%I:%M:%S %p")
                    today = now.strftime("%Y-%m-%d")

                    car = 'car'
                    bus = '--'
                    truck = '--'
                    amount = 10

                    c.execute("INSERT INTO toll(datestamp, time, car,bus,truck,amount) VALUES (?,?, ?, ?, ?,?)",
                              (today, ctime, car, bus, truck, amount))

                    conn.commit()
                else : continue
                ########################### CAR COUNT #####################################

                v = """SELECT count(car) FROM toll GROUP BY car"""
                c.execute(v)
                record = c.fetchall()
                car_count_list = []
                for row in record:
                    car_count_list = row[0]
                print("Printing Car  Count", car_count_list)

                ############################# BUS COUNT ##################################

                sel = """SELECT count(bus) FROM toll GROUP BY bus"""
                c.execute(sel)
                record = c.fetchall()
                count_list = []
                for row in record:
                    count_list = row[0]
                print("Printing Bus Count", count_list)
                ############################## TRUCK COUNT ###############################

                sel = """SELECT count(truck) FROM toll GROUP BY truck"""
                c.execute(sel)
                record = c.fetchall()
                print("size", record)
                global truck_count
                sel_truck = """SELECT truck FROM toll  """
                c.execute(sel_truck)
                rec = c.fetchall()
                for r in rec :
                    print(r[0])
                conn.commit()
                ############################## TWO COUNT ###############################
                sel_two = """SELECT count(truck) FROM toll WHERE truck = '2 axle' """
                c.execute(sel_two)
                rec_two = c.fetchall()
                for row in rec_two:
                    print(row)
                flag = 3
                conn.commit()
                ############################## THREE COUNT ###############################
                sel_three = """SELECT truck FROM toll WHERE truck = '3 axle' """
                c.execute(sel_three)
                rec_three = c.fetchall()
                for row1 in rec_three:
                    print(row1)
                flag = 4
                conn.commit()
                ############################## FOUR COUNT ###############################


                if flag == 2:
                    c = conn.cursor()
                    now = datetime.datetime.now()
                    today = now.strftime("%Y-%m-%d")

                    car = car_count_list

                    # c.execute("INSERT INTO count(eight_above,seven,six,five,four,three,two,car,bus) VALUES "
                    # "(?, ?, ?, ?,?,?,?,?,?)", (eight_or_more, seven, six, five, four, three, two, car, bus))



                    up = """UPDATE count SET date =? ,car=? WHERE count_id =1"""

                    val = (today, car)
                    c.execute(up, val)

                    global car_sum
                    car_sum = car * 10
                    up = """UPDATE amount SET date =?, car_sum=? WHERE amt_id =1"""
                    val = (today, car_sum)
                    c.execute(up, val)

                    conn.commit()
                elif flag == 1:

                    c = conn.cursor()

                    now = datetime.datetime.now()
                    today = now.strftime("%Y-%m-%d")

                    bus = count_list

                    #c.execute("INSERT INTO count (date,bus) VALUES  (?, ?)", (today,bus))

                    up = """UPDATE count SET date =?, bus=? WHERE count_id =1"""

                    val = (today, bus)
                    c.execute(up, val)
                    global bus_sum
                    bus_sum = bus * 20
                    up1 = """UPDATE amount SET date =?, bus_sum=? WHERE amt_id =1"""
                    val1 = (today, bus_sum)
                    c.execute(up1,val1)

                    conn.commit()
                elif flag == 3:

                    c = conn.cursor()

                    now = datetime.datetime.now()
                    today = now.strftime("%Y-%m-%d")
                    tr_2count = row
                    up1 = """UPDATE count SET date =?, 2axle=? WHERE count_id =1"""

                    val1 = (today, tr_2count)
                    c.execute(up1, val1)

                    global truck_sum
                    truck_sum = tr_2count * 50
                    up2 = """UPDATE amount SET date =?, truck_sum=? WHERE amt_id =1"""
                    val2 = (today, truck_sum)
                    c.execute(up2, val2)

                elif flag == 4:

                    c = conn.cursor()

                    now = datetime.datetime.now()
                    today = now.strftime("%Y-%m-%d")
                    tr_3count = row
                    up1 = """UPDATE count SET date =?, 3axle=? WHERE count_id =1"""

                    val1 = (today, tr_3count)
                    c.execute(up1, val1)

                    global truck_sum
                    truck_sum = tr_3count * 50
                    up2 = """UPDATE amount SET date =?, truck_sum=? WHERE amt_id =1"""
                    val2 = (today, truck_sum)
                    c.execute(up2, val2)

                    conn.commit()


            ####################################################MRCNN END##################################################

        end = timer()

        with open('data.json', 'w') as outfile:
            json.dump(dict2, outfile)

        print(end - start)
        return image,out_classes

    def close_session(self):
        self.sess.close()


