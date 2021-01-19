import os
import cv2
import json
import pickle
import shutil
import numpy as np
import tensorflow as tf
from test import test_yolo_model

def load_pickle(filename):
    with open(filename , "rb") as content:
        return pickle.load(content)

def write_bb_files(bb_dict , is_gt):
    folder_name = "ground_truth/" if is_gt else "predictions/"
    write_dir = "./data/{}".format(folder_name)
    if os.path.exists(write_dir):
        print("Deleting existing directory : " , write_dir)
        shutil.rmtree(write_dir)
    os.mkdir(write_dir)

    for inst in bb_dict:
        filename = inst
        f = open("{}{}.txt".format(write_dir, filename) , "w")
        b_boxes = bb_dict[inst]
        for i , object in enumerate(b_boxes):
            obj = []
            if not is_gt:
                obj = object
            else:
                obj += ["FACE"]
                obj += object

            obj = map(str , obj)
            obj = " ".join(obj)
            f.write(obj + "\n")
        f.close()

def transform_ground_truths(gt_dict , new_height = 416 , new_width = 416):
    trans_gt = {}
    image_names = []
    for instance in gt_dict:
        image_names.append(instance)
        img_name = instance.split("/")[-1].split(".")[0]
        objects = gt_dict[instance]

        rescaled_objects = []
        for object in objects:
            X = int(object[0] * new_width)
            Y = int(object[1] * new_height)
            W = int(object[2] * new_width)
            H = int(object[3] * new_height)
            rescaled_objects.append([X , Y , W , H])

        trans_gt.update({img_name : rescaled_objects})

    return trans_gt , image_names

def evaluate(anchors , test_path , image_names_path , chkpnt_dir , conv_weights_file , bn_weights_file , val_datapath , gt_path , pred_path , total_images, show_output = True , eval_map = False , session = None):
    if not eval_map:
        raise Exception("eval_map has to be true to calculate mAP metric for statistics.")

    batch_size = 50
    start_index = 0
    test_data = load_pickle(test_path)
    print("*******" , len(test_data) , "*********")
    image_names = load_pickle(image_names_path)
    test_yolo_model(test_data , total_images , start_index , chkpnt_dir , anchors , conv_weights_file , bn_weights_file , val_datapath , image_names = image_names , show_output = show_output , eval_map = eval_map ,sess = session)

    ground_truth = load_pickle(gt_path)
    predictions = load_pickle(pred_path)
    trans_gt , filtered_image_names = transform_ground_truths(ground_truth)
    compare_predictions = False
    filtered_gt = {}

    for key in predictions:
        filtered_gt.update({key : trans_gt[key]})

    if compare_predictions:
        for key in filtered_gt.keys():
            print("*******************************")
            print(key)
            gt = filtered_gt[key]
            preds = predictions[key]
            for i in range(len(filtered_image_names)):
                if key in filtered_image_names[i].split("/")[-1]:
                    image = cv2.resize(cv2.imread(filtered_image_names[i] , 1) , (416,416))
                    for object in gt:
                        cv2.rectangle(image , (int(object[0]) , int(object[1])) , (int(object[0]) + int(object[2]) , int(object[1]) + int(object[3])) , (0 , 0 , 255) , 2)

                    for object in preds:
                        cv2.rectangle(image , (int(object[2]) , int(object[3])) , (int(object[2]) + int(object[4]) , int(object[3]) + int(object[5])) , (0 ,255 , 0) , 2)

                    cv2.imshow(key , image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    write_bb_files(filtered_gt , True)
    write_bb_files(predictions , False)

    curr_dir = os.getcwd()
    gt_dir = curr_dir + "/data/ground_truth/"
    pred_dir = curr_dir + "/data/predictions/"
    coord_format = "xywh"
    os.system("python ./Object-Detection-Metrics/pascalvoc.py -gt {} -det {} -gtformat {} -detformat {} -t 0.4".format(gt_dir, pred_dir, coord_format ,coord_format))


if __name__ == "__main__":
    config_path = "./config.json"
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    gt_path = config["evaluation_details"]["gt_path"]
    pred_path = config["evaluation_details"]["pred_path"]
    test_path = config["inference_details"]["test_path"]
    image_names_path = config["inference_details"]["image_names_path"]
    chkpnt_dir = config["inference_details"]["chkpnt_dir"]
    total_images = config["evaluation_details"]["num_images"]
    start_index = config["evaluation_details"]["start_index"]
    conv_weights_file = config["train_details"]["pre_trained_conv_weights"]
    bn_weights_file = config["train_details"]["pre_trained_bn_weights"]
    val_datapath = config["preprocessing"]["val_datapath"]
    anchors = config["model"]["anchors"]

    anchors = np.array(anchors).astype(np.float32) * 13
    show_output = config["evaluation_details"]["show_output"]
    eval_map = config["evaluation_details"]["eval_map"]
    evaluate(anchors , test_path , image_names_path , chkpnt_dir , conv_weights_file , bn_weights_file , val_datapath , gt_path , pred_path , total_images, show_output , eval_map)
