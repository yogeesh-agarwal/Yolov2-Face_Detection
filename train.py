import os
import sys
import json
import utils
import numpy as np
import tensorflow as tf
from model import Yolov2
from loss import Yolov2Loss
from evaluate_mAP import evaluate
from pre_processing import DataGenerator
from tensorflow.python import debug as tf_debug

def save(session , saver , checkpoint_dir , step):
    dir = os.path.join(checkpoint_dir , "yolov2")
    saver.save(session , dir , global_step = step)
    print("model saved at {} for epoch {}".format(dir , step))

def load_model(session , saver , checkpoint_dir):
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        epoch_no = int(ckpt_name[7 : ])
        print("ckpt_name : " , ckpt_name)
        ckpt_file = os.path.join(checkpoint_dir , ckpt_name)
        print("checkpoint_file : " , ckpt_file)
        saver.restore(session , ckpt_file)
        return (True , epoch_no)
    else:
        return (False , 0)

def train(input_size ,
          grid_size ,
          batch_size,
          num_epochs,
          init_lr,
          anchors ,
          logs_dir,
          save_dir,
          conv_weights_file,
          bn_weights_file,
          data_path ,
          train_file ,
          image_names ,
          gt_path ,
          pred_path ,
          test_path ,
          val_image_names_path ,
          chkpnt_dir ,
          val_total_images ,
          val_start_index ,
          labels ,
          is_norm,
          show_output,
          eval_map):

    input = tf.placeholder(dtype = tf.float32 , shape = [None , input_size , input_size , 3])
    ground_truth = tf.placeholder(dtype = tf.float32 , shape = [None , grid_size , grid_size , len(anchors) , (4 + 1 + len(labels))])
    is_training = tf.placeholder(dtype = tf.bool , shape = [])
    g_step = tf.get_variable('global_step', trainable=False, initializer=0)

    data_generator = DataGenerator(input_size , grid_size , anchors , data_path , train_file , image_names , labels , is_norm, True)
    model = Yolov2(conv_weights_file , bn_weights_file , grid_size , len(anchors) , len(labels) , is_training)
    predictions = model.gen_model(input)
    yolo_loss = Yolov2Loss(batch_size , grid_size , grid_size , len(anchors) , len(labels) , anchors)
    loss , merged_summary_op = yolo_loss.get_loss(ground_truth , predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate = init_lr , name = "optimizer")
    train_step = optimizer.minimize(loss)

    num_iter = min(data_generator.num_train_instances , 5000) // batch_size
    summary_writer = tf.summary.FileWriter(logs_dir , graph = tf.get_default_graph())
    file = open("losses.txt" , "a")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver(max_to_keep = 4)
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess , "yogeesh-ubuntu:7000")
        if_checkpoint , epoch_start = load_model(sess , saver , save_dir)
        if if_checkpoint:
            print("Loading partially trained model at epoch : " ,epoch_start)
        else:
            print("training from scratch")

        curr_max_loss = 1e+9
        for epoch in range(epoch_start , num_epochs):
            iter_loss = 0
            for iter in range(num_iter):
                current_image_batch , current_label_batch , best_anchor_indexes = data_generator.load_data(batch_size , iter)
                current_loss , _ , summary_opr = sess.run([loss , train_step , merged_summary_op] , feed_dict = {input : current_image_batch , ground_truth : current_label_batch , is_training : True})
                iter_loss += current_loss
                summary_writer.add_summary(summary_opr , epoch)

            current_gstep = tf.train.global_step(sess, g_step)
            per_epoch_loss = iter_loss / num_iter
            loss_string = "epoch_no : {} , current_loss : {}".format(epoch , per_epoch_loss)
            file.write(loss_string + "\n")

            if(per_epoch_loss >= 0.01 and per_epoch_loss <= 0.06):
                save(sess , saver , save_dir , epoch)
                print("stopping early" , per_epoch_loss)
                sys.exit(0)

            if(epoch % 5 == 0):
                # evaluate(anchors , test_path , val_image_names_path , chkpnt_dir , gt_path , pred_path , val_total_images, show_output , eval_map , sess)
                print(loss_string)
                if(per_epoch_loss < curr_max_loss):
                    curr_max_loss = per_epoch_loss
                    save(sess , saver , save_dir , epoch)

    print("Training Completed!!!")
    file.close()



def main(config):
    # anchors are expected to be in format as [width , height] and in grid_cell [0 ,13] unit.
    gt_path = config["train_details"]["gt_path"]
    labels = config["model"]["classes"]
    is_norm = config["train_details"]["is_norm"]
    anchors = config["model"]["anchors"]
    init_lr = config["train_details"]["init_lr"]
    logs_dir = config["train_details"]["logs_dir"]
    save_dir = config["train_details"]["save_dir"]
    pred_path = config["train_details"]["pred_path"]
    test_path = config["train_details"]["test_path"]
    grid_size = config["train_details"]["grid_size"]
    batch_size = config["train_details"]["batch_size"]
    train_file = config["train_details"]["train_file"]
    num_epochs = config["train_details"]["num_epochs"]
    chkpnt_dir = config["train_details"]["chkpnt_dir"]
    input_size = config["model"]["input_shape"]
    image_names = config["train_details"]["image_names"]
    data_path = config["train_details"]["train_data_path"]
    eval_map = config["evaluation_details"]["eval_map"]
    bn_weights_file = config["train_details"]["pre_trained_bn_weights"]
    show_output = config["evaluation_details"]["show_output"]
    val_image_names_path = config["train_details"]["val_image_names_path"]
    conv_weights_file = config["train_details"]["pre_trained_conv_weights"]
    anchors = np.array(anchors).astype(np.float32) * (input_size / (input_size / grid_size))

    val_total_images = 700
    val_start_index = 0

    train(input_size ,
          grid_size ,
          batch_size  ,
          num_epochs ,
          init_lr ,
          anchors ,
          logs_dir ,
          save_dir ,
          conv_weights_file ,
          bn_weights_file ,
          data_path ,
          train_file ,
          image_names ,
          gt_path ,
          pred_path ,
          test_path ,
          val_image_names_path ,
          chkpnt_dir ,
          val_total_images ,
          val_start_index ,
          labels ,
          is_norm ,
          show_output ,
          eval_map)

if __name__ == "__main__":
    config_path = "./config.json"
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    main(config)
