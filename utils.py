import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self , x , y , w , h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_area(self):
        return self.w * self.h

    def convert2xyxy(self):
        x2 = self.x + self.w
        y2 = self.y + self.h
        return np.array([self.x , self.y , x2 , y2])

    def iou(self , other):
        transformed_box1 = self.convert2xyxy()
        transformed_box2 = other.convert2xyxy()

        intersect_width = None
        if transformed_box2[0] < transformed_box1[0]:
            if transformed_box2[2] < transformed_box1[0]:
                intersect_width = 0
            else:
                intersect_width = min(transformed_box1[2] , transformed_box2[2]) - transformed_box1[0]
        else:
            if transformed_box2[0] > transformed_box1[2]:
                intersect_width = 0
            else:
                intersect_width = min(transformed_box1[2] , transformed_box2[2]) - transformed_box2[0]

        intersect_height = None
        if transformed_box2[1] < transformed_box1[1]:
            if transformed_box2[3] < transformed_box1[1]:
                intersect_height = 0
            else:
                intersect_height = min(transformed_box1[3] , transformed_box2[3]) - transformed_box1[1]
        else:
            if transformed_box2[1] > transformed_box1[3]:
                intersect_height = 0
            else:
                intersect_height = min(transformed_box1[3] , transformed_box2[3]) - transformed_box2[1]

        intersect_area = intersect_height * intersect_width
        union_area = self.get_area() + other.get_area() - intersect_area

        return float(intersect_area) / union_area

    def __repr__(self):
        return "center_X = {} , center_Y = {} , width = {} , height = {}".format(self.x , self.y , self.w , self.h)

def gen_cell_grid(grid_w , grid_h , num_anchors):
    cell_grid = np.zeros((grid_h , grid_w , num_anchors , 2) , dtype = np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            for anc_index in range(num_anchors):
                cell_grid[row , col , anc_index , 0] = col
                cell_grid[row , col , anc_index , 1] = row

    return cell_grid

def gen_anchor_grid(grid_w , grid_h , num_anchors , anchors):
    anchor_grid = np.ones([grid_h , grid_w , num_anchors , 2] , dtype = np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            for anc_index in range(num_anchors):
                anchor_grid[row , col , anc_index , 0] = anchors[anc_index , 0]
                anchor_grid[row , col , anc_index , 1] = anchors[anc_index , 1]

    return anchor_grid

def transform_bbox(coord_xy , coord_wh , image_width , image_height):
    zero = tf.constant(0 , dtype = tf.float32)
    image_shape = tf.constant([image_width - 1, image_height - 1] , dtype = tf.float32)
    coord_mins = coord_xy - (coord_wh / 2)
    coord_mins = tf.minimum(tf.maximum(zero , coord_mins) , image_shape)
    coord_maxs = coord_xy + (coord_wh / 2)
    coord_maxs = tf.maximum(tf.minimum(coord_maxs, image_shape) , zero)
    return coord_mins , coord_maxs

def loss_iou(gt_xy , gt_wh , pred_xy , pred_wh , image_height , image_width , giou = False):
    zero = tf.convert_to_tensor(0.0 , tf.float32)
    gt_mins , gt_maxs = transform_bbox(gt_xy , gt_wh , image_width , image_height)
    pred_mins , pred_maxs = transform_bbox(pred_xy , pred_wh , image_width , image_height)

    Xmin = tf.maximum(gt_mins[... , 0] , pred_mins[... , 0])
    Ymin = tf.maximum(gt_mins[... , 1] , pred_mins[... , 1])
    Xmax = tf.minimum(gt_maxs[... , 0] , pred_maxs[... , 0])
    Ymax = tf.minimum(gt_maxs[... , 1] , pred_maxs[... , 1])

    intersection_width = tf.maximum((Xmax - Xmin) , 0.0)
    intersection_height = tf.maximum((Ymax - Ymin) , 0.0)
    intersection_area = intersection_height * intersection_width

    union_area = ((gt_wh[... , 0] * gt_wh[... , 1]) + (pred_wh[... , 0] * pred_wh[... , 1])) - intersection_area + 1e-8
    iou = tf.math.truediv(intersection_area, union_area)

    if giou:
        convex_mins = tf.minimum(gt_mins , pred_mins)
        convex_maxs = tf.maximum(gt_maxs , pred_maxs)
        convex_wh = tf.maximum(zero , convex_maxs - convex_mins)
        convex_area = convex_wh[... , 0] * convex_wh[... , 1] + 1e-16
        # giou =  iou - tf.divide_no_nan((convex_area - union_area) / convex_area)
        giou =  iou - tf.math.truediv((convex_area - union_area) , convex_area)
        return giou

    return iou
