import os
import cv2
import math
import utils
import pickle
import random
import numpy as np
import imgaug as ia
from operator import itemgetter
import imgaug.augmenters as iaa

class DataGenerator:
    def __init__(self , input_size , grid_size , anchors , data_path , train_file , image_names , labels , is_norm , is_augment , test_file = None , val_file = None ):
        self.labels = labels
        self.anchors = anchors
        self.is_norm = is_norm
        self.data_path = data_path
        self.grid_width = grid_size
        self.grid_height = grid_size
        self.input_width = input_size
        self.input_height = input_size
        self.is_augment = is_augment
        self.val_data = None
        self.test_data = None
        if val_file:
            self.val_data = self.load_pickle(val_file)
            self.num_val_instances = len(self.val_data)
        self.train_data = self.load_pickle(train_file)
        self.images_names = self.load_pickle(image_names)
        self.num_train_instances = len(self.train_data)

        sometimes = lambda aug : iaa.Sometimes(0.55, aug)
        self.augmentor = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            sometimes(iaa.Affine(
                rotate = (-20 , 20),
                shear = (-13 , 13),
            )),
            iaa.SomeOf((0 , 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0 , 3.0)),
                    iaa.AverageBlur(k =  (2 , 7)),
                    iaa.MedianBlur(k = (3 , 11)),
                ]),
                iaa.Sharpen(alpha = (0 , 1.0) , lightness = (0.75 , 1.5)),
                iaa.AdditiveGaussianNoise(loc = 0 , scale = (0.0 , 0.05*255) , per_channel = 0.5),
                iaa.Add((-10 , 10) , per_channel = 0.5),
                iaa.Multiply((0.5 , 1.5) , per_channel = 0.5),
                iaa.LinearContrast((0.5 , 2.0) , per_channel = 0.5),
            ] , random_order = True)
        ] , random_order = True)


    def load_pickle(self , filepath):
        with open(filepath , "rb") as content:
            return pickle.load(content)

    def augment_data(self , images , labels):
        bb_objects = []
        for objects in labels:
            curr_bb_objects = []
            for object in objects:
                curr_bb_objects.append(ia.BoundingBox(object[0] , object[1] , object[0]  + object[2] , object[1] + object[3]))
            bb_objects.append(curr_bb_objects)

        aug_images , aug_bbs = self.augmentor(images = images , bounding_boxes = bb_objects)
        augmented_boxes = []
        augmented_images = []

        for image , bb in zip(aug_images , aug_bbs):
            boxes = []
            for bbox in bb:
                boxes.append([bbox.x1 , bbox.y1 , (bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)])
            augmented_boxes.append(boxes)
            augmented_images.append(image)
        return augmented_images , augmented_boxes

    def encode_data(self, starting_index , ending_index):
        images = []
        labels = []
        for index in range(starting_index , ending_index):
            image_path = os.path.join(self.data_path , self.images_names[index])
            image = cv2.cvtColor(cv2.imread(image_path) , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image , (self.input_width , self.input_height))
            if self.images_names[index] not in self.train_data:
                raise Exception("{} not found in training data".format(sefl.images_names[index]))
            label = self.train_data[self.images_names[index]]  #contains list of objects in one image.

            # resizing the boxes according to the input width and height.
            resized_label = []
            for index in range(len(label)):
                X = int(label[index][0] * self.input_width)
                Y = int(label[index][1] * self.input_height)
                W = int(label[index][2] * self.input_width)
                H = int(label[index][3] * self.input_height)
                resized_label.append([X , Y , W , H])
            images.append(image)
            labels.append(resized_label)

        augmented_images , augmented_boxes = images , labels
        if self.is_augment:
            augmented_images , augmented_boxes = self.augment_data(images , labels)
        encoded_images = []
        encoded_labels = []
        best_anchors_indexes = []
        for index in range(len(augmented_images)):
            aug_image = augmented_images[index]
            if self.is_norm:
                aug_image = aug_image / 255.

            detector_indexes = []
            e_label = np.zeros([self.grid_height , self.grid_width , len(self.anchors) , (4+1+len(self.labels))])
            for object in augmented_boxes[index]:
                center_x = object[0] + (object[2]*0.5)
                center_y = object[1] + (object[3]*0.5)
                center_x = center_x / (float(self.input_width) / self.grid_width) # in the range (0 , grid_width)
                center_y = center_y / (float(self.input_height) / self.grid_height) # in the range (0 , grid_height)
                center_w = object[2] / (float(self.input_width) / self.grid_width)
                center_h = object[3] / (float(self.input_height) / self.grid_height)

                max_iou = -1
                best_anchor_index = 0
                dummy_box = utils.BoundingBox(0 , 0 , center_w , center_h)
                for index in range(len(self.anchors)):
                    anchor_box = utils.BoundingBox(0 , 0 , self.anchors[index][0] , self.anchors[index][1])
                    iou = dummy_box.iou(anchor_box)
                    if iou > max_iou:
                        max_iou = iou
                        best_anchor_index = index

                grid_x = min(int(math.floor(center_x)) , self.grid_width - 1)
                grid_y = min(int(math.floor(center_y)) , self.grid_height - 1)
                e_label[grid_y , grid_x , best_anchor_index , 0:4] = [center_x , center_y , center_w , center_h]  #bounding box
                e_label[grid_y , grid_x , best_anchor_index , 4] = 1   #objectness
                e_label[grid_y , grid_x , best_anchor_index , 5] = 1   #class probs
                e_label[grid_y , grid_x , best_anchor_index , 6] = 0
                detector_indexes.append(best_anchor_index)

            encoded_images.append(aug_image)
            encoded_labels.append(e_label)
            best_anchors_indexes.append(detector_indexes)

        return encoded_images , encoded_labels , best_anchors_indexes

    def load_data(self , batch_size , index):
        starting_index = index * batch_size
        ending_index = starting_index + batch_size
        if ending_index > len(self.train_data):
            ending_index = len(self.train_data)
            starting_index = ending_index - batch_size

        index_list = list(range(batch_size))
        random.shuffle(index_list)
        encoded_images , encoded_labels , best_anchors_indexes = self.encode_data(starting_index , ending_index)
        encoded_images = list(itemgetter(*index_list)(encoded_images))
        encoded_labels = list(itemgetter(*index_list)(encoded_labels))
        best_anchors_indexes = list(itemgetter(*index_list)(best_anchors_indexes))


        image_data = np.array(encoded_images).reshape(batch_size , self.input_height , self.input_width , 3)
        label_data = np.array(encoded_labels).reshape(batch_size , self.grid_height , self.grid_width , len(self.anchors) , (4+1+len(self.labels)))
        return image_data , label_data , best_anchors_indexes
