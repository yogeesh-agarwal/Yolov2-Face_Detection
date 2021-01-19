import tensorflow as tf
import numpy as np
import utils
import sys

class Yolov2Loss():
    def __init__(self, batch_size , grid_h , grid_w , num_anchors , num_classes , anchors):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.anchors = anchors
        self.epsilon = 1e-6
        self.obj_scale = 1.0
        self.coord_scale = 5.0
        self.noobj_scale = 0.5
        self.class_scale = 1.0
        self.iou_treshold = 0.6
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_height = 416
        self.input_width = 416
        self.box_adjuster = 4

    def modify_predictions(self):
        cell_grid = utils.gen_cell_grid(self.grid_w , self.grid_h , self.num_anchors)
        anchor_grid = utils.gen_anchor_grid(self.grid_w , self.grid_h , self.num_anchors , self.anchors)
        pred_xy = tf.sigmoid(self.predictions[... , 0:2]) + cell_grid
        pred_wh = tf.nn.sigmoid(self.predictions[... , 2:4]) * anchor_grid * self.box_adjuster
        pred_conf = tf.sigmoid(self.predictions[..., 4:5])
        pred_class = self.predictions[... , 5:]
        return pred_xy , pred_wh , pred_conf , pred_class

    def modify_gt(self , pred_xy , pred_wh):
        gt_xy = self.ground_truth[... , 0:2]
        gt_wh = self.ground_truth[... , 2:4]
        ious = utils.loss_iou(gt_xy, gt_wh , pred_xy , pred_wh , self.input_height , self.input_width)
        gt_conf = tf.expand_dims(ious , axis = -1) * self.ground_truth[... , 4:5]
        gt_class = tf.argmax(self.ground_truth[... , 5:] , -1)
        return gt_xy , gt_wh , gt_conf , gt_class

    def gen_masks(self , gt_conf , pred_xy , pred_wh):
        condition = tf.greater(gt_conf , tf.constant(0.0 , dtype = tf.float32))
        mask = tf.where(condition , tf.ones_like(gt_conf) , tf.zeros_like(gt_conf))
        total_nb = tf.reduce_sum(tf.to_float(mask > 0.0))
        coord_mask = tf.multiply(mask , self.coord_scale)

        mask_shape = [self.batch_size , self.grid_h , self.grid_w, self.num_anchors , 1]
        gt_true_boxes = tf.reshape(self.ground_truth[... , 0:4] , [self.batch_size , 1 , 1, 1 , (self.grid_h * self.grid_w * self.num_anchors) , 4])
        gt_xy = gt_true_boxes[... , 0:2]
        gt_wh = gt_true_boxes[... , 2:4]
        pred_boxes_xy = tf.expand_dims(pred_xy , axis = 4)
        pred_boxes_wh = tf.expand_dims(pred_wh , axis = 4)
        potential_best_ious = utils.loss_iou(gt_xy , gt_wh , pred_boxes_xy , pred_boxes_wh , self.input_height  ,self.input_width)
        best_ious = tf.reduce_max(potential_best_ious , axis = 4)
        conf_mask = tf.zeros(mask_shape)
        no_object_mask = conf_mask + tf.expand_dims(tf.to_float(best_ious < self.iou_treshold) * (1 - self.ground_truth[... , 4]) * self.noobj_scale , axis = -1)
        object_mask = tf.expand_dims(self.ground_truth[... , 4] * self.obj_scale , axis = -1)

        conf_mask = no_object_mask + object_mask
        class_mask = tf.expand_dims(self.ground_truth[... , 4] * self.class_scale , axis = -1)

        return coord_mask , conf_mask , class_mask  , total_nb

    def localization_loss(self , mask , total_nb , gt_xy , gt_wh , pred_xy,  pred_wh):
        xy_loss = tf.reduce_sum(tf.multiply(mask , tf.square(gt_xy - pred_xy))) / (total_nb + self.epsilon) / 2.
        wh_loss = tf.reduce_sum(tf.multiply(mask , tf.square(gt_wh - pred_wh))) / (total_nb + self.epsilon) / 2.
        tf.summary.scalar("loss_xy" , xy_loss)
        tf.summary.scalar("loss_wh" , wh_loss)
        coord_loss = xy_loss + wh_loss
        coord_loss = tf.multiply(coord_loss , self.coord_scale)
        tf.summary.scalar("coord_loss", coord_loss)
        return xy_loss , wh_loss , coord_loss

    def giou_loss(self , num_coord_objects , gt_xy , gt_wh, pred_xy , pred_wh):
        giou = utils.loss_iou(gt_xy , gt_wh , pred_xy , pred_wh , giou = True)
        g_loss = (tf.reduce_sum(giou) * self.coord_scale) /(num_coord_objects + self.epsilon)
        return g_loss

    def conf_loss(self , conf_mask , total_nb , gt_conf , pred_conf):
        loss_conf = tf.reduce_sum(tf.square(pred_conf - gt_conf) * conf_mask , name = "loss_conf") / (total_nb + self.epsilon) / 2.
        tf.summary.scalar("conf_loss" , loss_conf)
        return loss_conf

    def class_loss(self , mask , total_nb , gt_class , pred_class):
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt_class, logits = pred_class)
        loss_class = tf.reduce_sum(tf.multiply(mask , tf.expand_dims(loss_class , axis = -1)))
        loss_class = tf.multiply(loss_class , self.class_scale)
        tf.summary.scalar("class_loss" , loss_class)
        return loss_class

    def get_loss(self , ground_truth , predictions):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.shape = tf.shape(ground_truth)
        pred_xy , pred_wh , pred_conf , pred_class = self.modify_predictions()
        gt_xy , gt_wh , gt_conf , gt_class = self.modify_gt(pred_xy , pred_wh)
        coord_mask , conf_mask , class_mask , total_nb = self.gen_masks(gt_conf , pred_xy , pred_wh)

        loss_xy , loss_wh , coord_loss = self.localization_loss(coord_mask , total_nb , gt_xy , gt_wh , pred_xy , pred_wh)
        conf_loss = self.conf_loss(conf_mask , total_nb , gt_conf , pred_conf)
        class_loss = self.class_loss(class_mask , total_nb , gt_class , pred_class)
        tf.summary.scalar("class_loss :" , class_loss)

        print_op  = tf.print("loss_xy , loss_wh , conf_loss , class_loss : " , loss_xy , loss_wh , conf_loss , class_loss , output_stream = sys.stdout)
        with tf.control_dependencies([print_op]):
            total_loss = coord_loss + conf_loss + class_loss
            tf.summary.scalar("total loss : " , total_loss)
            merged_summary_op = tf.summary.merge_all()

        return total_loss , merged_summary_op
