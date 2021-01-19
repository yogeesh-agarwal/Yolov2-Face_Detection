import os
import sys
import cv2
import json
import time
import utils
import pickle
import argparse
import numpy as np
import tensorflow as tf
from model import Yolov2
import post_processing as post_process

def load_pickle(filepath):
    with open(filepath , "rb") as content:
        return pickle.load(content)


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

def get_test_data(test_data , val_datapath , num_images , start_index , end_index):
    count = 0
    test_images = []
    for image_name in test_data:
        if count >= start_index and count < end_index:
            image_path = os.path.join(val_datapath , image_name)
            img = cv2.imread(image_path , 1)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            img = cv2.resize(img , (416,416))
            test_images.append(img)
        count += 1
    return test_images

def normalize(images):
    normalized_images = []
    for image in images:
        normalized_images.append(image / 255.0)
    return normalized_images

def RGB_to_BGR(images):
    color_converted_images = []
    for img in images:
        color_converted_images.append(np.array(img[:, : , ::-1]))

    return color_converted_images

def test_webcam_input(chkpnt_dir , anchors , conv_weights_file , bn_weights_file , image_name = None , show_output = True , eval_map = False , sess = None):
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [1,416,416,3])
    is_training = tf.placeholder(dtype = tf.bool)
    model = Yolov2(conv_weights_file , bn_weights_file , 13 , len(anchors) , 2 , is_training)
    predictions = model.gen_model(tf_input)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    pred_dict = None
    if sess == None:
        sess = tf.Session(config = config)

    saver = tf.train.Saver()
    checkpoint = load_model(sess , saver , chkpnt_dir)
    if not checkpoint:
        print("No trained model predsent")

    print("press q to quit the display window")
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        stime = time.time()
        ret , frame = cap.read()
        preprocessed_frame = np.array(cv2.resize(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) , (416,416)) , dtype = np.float32)
        inp_frame = preprocessed_frame.reshape(1,416,416,3) / 255.0
        prediction = sess.run(predictions , feed_dict = {tf_input : inp_frame , is_training : False})
        if ret:
            post_process.post_process([cv2.resize(frame , (416,416))] , prediction , anchors , pred_dict , image_name , show_output , eval_map , "camera")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print('FPS {:1f}'.format(1/(time.time() - stime)))
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def test_video_file(video_file_path , chkpnt_dir , anchors , conv_weights_file , bn_weights_file , image_name = None , show_output = True , eval_map = False , sess = None):

    tf_input = tf.placeholder(dtype = tf.float32 , shape = [1,416,416,3])
    is_training = tf.placeholder(dtype = tf.bool)
    model = Yolov2(conv_weights_file , bn_weights_file , 13 , len(anchors) , 2 , is_training)
    predictions = model.gen_model(tf_input)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    pred_dict = None
    if sess == None:
        sess = tf.Session(config = config)

    saver = tf.train.Saver()
    checkpoint = load_model(sess , saver , chkpnt_dir)
    if not checkpoint:
        print("No trained model present")

    print("press q to quit the display window")
    cap = cv2.VideoCapture(video_file_path)
    while(cap.isOpened()):
        ret , frame = cap.read()
        preprocessed_frame = np.array(cv2.resize(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) , (416,416)) , dtype = np.float32)
        inp_frame = preprocessed_frame.reshape(1,416,416,3) / 255.0
        prediction = sess.run(predictions , feed_dict = {tf_input : inp_frame , is_training : False})
        if ret:
            post_process.post_process([cv2.resize(frame , (416,416))] , prediction , anchors , pred_dict , image_name , show_output , eval_map , "video")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def test_yolo_model(test_data , num_images , start_index , chkpnt_dir , anchors , conv_weights_file , bn_weights_file , val_datapath , image_names = None , show_output = True , eval_map = False , sess = None):
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [None,416,416,3])
    is_training = tf.placeholder(dtype = tf.bool)
    model = Yolov2(conv_weights_file , bn_weights_file , 13 , len(anchors) , 2 , is_training)
    predictions = model.gen_model(tf_input)

    batch_size = 50
    num_batches = max(1 , num_images // batch_size)
    pred_dict = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if sess == None:
        sess = tf.Session(config = config)

    saver = tf.train.Saver()
    checkpoint = load_model(sess , saver , chkpnt_dir)
    if not checkpoint:
        print("No trained model present")

    for batch in range(num_batches):
        current_start_index = start_index + (batch * batch_size)
        current_ending_index = min(current_start_index + batch_size , num_images)
        print(current_start_index , current_ending_index)
        images = get_test_data(test_data , val_datapath , batch_size , current_start_index , current_ending_index)
        names = image_names[current_start_index : current_ending_index]

        prediction = sess.run(predictions , feed_dict = {tf_input : normalize(images) , is_training : False})
        pred_dict = post_process.post_process(RGB_to_BGR(images) , prediction , anchors , pred_dict , names , show_output , eval_map)

    if eval_map:
        with open("./data/val_predictions.pickle" , "wb") as f:
            pickle.dump(pred_dict , f , pickle.HIGHEST_PROTOCOL)

def test_single_image(img_path , chkpnt_dir , anchors , conv_weights_file , bn_weights_file , image_names = None , show_output = True , eval_map = False , sess = None):
    test_img = cv2.resize(cv2.cvtColor(cv2.imread(img_path , 1) , cv2.COLOR_BGR2RGB) , (416,416))
    tf_input = tf.placeholder(dtype = tf.float32 , shape = [None,416,416,3])
    is_training = tf.placeholder(dtype = tf.bool)
    model = Yolov2(conv_weights_file , bn_weights_file , 13 , len(anchors) , 2 , is_training)
    predictions = model.gen_model(tf_input)
    pred_dict = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if sess == None:
        sess = tf.Session(config = config)

    saver = tf.train.Saver()
    checkpoint = load_model(sess , saver , chkpnt_dir)
    if not checkpoint:
        print("No trained model present")
    prediction = sess.run(predictions , feed_dict = {tf_input : normalize([test_img]) , is_training : False})
    pred_dict = post_process.post_process(RGB_to_BGR([test_img]) , prediction , anchors , pred_dict , image_names , show_output , eval_map)

if __name__ == "__main__":
    init_message = "Test suite for face detection\n default mode : testing from wider face dataset validation set\n camera mode : use webcam for live detection\n video mode : test a video file \n single : test a single image\n"
    parser = argparse.ArgumentParser(init_message)
    parser.add_argument("--mode" , type = str , default = "validation_set" , help = "mode for testing")
    parser.add_argument("--img_path" , type = str , default = "" , help = "path of image to be tested")
    parser.add_argument("--video_path" , type = str , default = "" , help = "path of video to be tested")
    args = parser.parse_args()

    config_path = "./config.json"
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    input_size = config["model"]["input_shape"]
    grid_size = config["train_details"]["grid_size"]
    anchors = config["model"]["anchors"]
    anchors = np.array(anchors).astype(np.float32) * (input_size / (input_size / grid_size))
    test_path = config["inference_details"]["test_path"]
    image_names_path = config["inference_details"]["image_names_path"]
    chkpnt_dir = config["inference_details"]["chkpnt_dir"]
    num_images = config["inference_details"]["num_images"]
    start_index = config["inference_details"]["start_index"]
    conv_weights = config["train_details"]["pre_trained_conv_weights"]
    bn_weights = config["train_details"]["pre_trained_bn_weights"]
    val_datapath = config["preprocessing"]["val_datapath"]
    test_data = load_pickle(test_path)
    image_names = load_pickle(image_names_path)

    if args.mode == "validation_set":
        test_yolo_model(test_data, num_images , start_index , chkpnt_dir , anchors , conv_weights , bn_weights , val_datapath = val_datapath , image_names = image_names , show_output = True)
    elif args.mode == "camera":
        test_webcam_input(chkpnt_dir , anchors , conv_weights , bn_weights)
    elif args.mode == "single":
        img_path = args.img_path
        if img_path == "":
            raise Exception("please provide the path of image to be tested for face detection.")
        test_single_image(img_path , chkpnt_dir , anchors , conv_weights ,  bn_weights , image_names = None , show_output = True)
    elif args.mode == "video":
        video_path = args.video_path
        if video_path == "":
            raise Exception("please provide the path of video to be tested for face detection , default video path : ./data/fv_2.mp4")
        test_video_file(video_path , chkpnt_dir , anchors , conv_weights , bn_weights)

    else:
        raise Exception("Invalid mode selected , please selected either camera or validation_set as modes for testing")

    print("try and try untill u succeed.")
