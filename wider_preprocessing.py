import os
import sys
import cv2
import math
import json
import pickle
import numpy as np

def cal_overlap_ratio(box_w , box_h , image_height ,image_width , threshold = 0.1):
    image_area = image_height * image_width
    box_area = box_h * box_w

    if  box_area / image_area < threshold:
        print("discarding image")
        return False
    print(image_area , box_area , box_area / image_area)
    return True

def read_files(filepath , data_dir):
    file = open(filepath , "r")
    lines = file.readlines()
    train_data = {}
    image_names = []
    count = 1
    num_faces = ["0\n", "1\n","2\n","3\n","4\n","5\n","6\n","7\n","8\n","9\n","10\n"]
    try:
        while count < len(lines):
            valid_file = False
            length = int(lines[count].split("\n")[0])
            if lines[count] in num_faces:
                if lines[count] == "0\n":
                    print("0 faces in the image")
                    raise Exception("Zero faces in the image , " , lines[count-1])

                valid_file = True
                filename = lines[count-1][:-1]
                event_class =  int(filename.split("--")[0])
                image = cv2.imread(os.path.join(data_dir, filename) , 1)

                if length == 0:
                    count += 3
                    continue
                count += 1
                boxes = []
                discard_image = False
                for bb in range(length):
                    box = lines[count + bb].strip()
                    box = list(map(int , box.split(" ")))
                    if not cal_overlap_ratio(box[2] , box[3] , image.shape[0] , image.shape[1] , 0.01):
                        discard_image = True
                        break
                    bbox = box[:4]
                    boxes.append(bbox)
                if not discard_image:
                    train_data[filename] = boxes
                    image_names.append(filename)
            count += (length + 1)
            if not valid_file:
                count += 1
    except Exception as e:
        print("error encontered : " , e)
        print(count , lines[count])
        print(filename , event_class)
        sys.exit(0)

    return train_data , image_names

def normalize_coords(train_data , data_path):
    for index , img_name in enumerate(train_data):
        image_path = os.path.join(data_path , img_name)
        objects = train_data[img_name]
        image = cv2.imread(image_path , 1)
        height , width = image.shape[0] , image.shape[1]
        norm_objects = []
        for object in objects:
            object[0] /= width
            object[1] /= height
            object[2] /= width
            object[3] /= height
            norm_objects.append(object)
        print("{} image objects are normalized with height = {} , width = {}".format(index+1 , height , width))
        train_data[img_name] = norm_objects
    return train_data

if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    train_data , image_names = read_files(config["preprocessing"]["train_data_filepath"] , config["preprocessing"]["train_datapath"])
    train_data = normalize_coords(train_data , config["preprocessing"]["train_datapath"])
    for index , img_path in enumerate(train_data):
        print(img_path)
        for object in train_data[img_path]:
            print(f"************{object}****************")

    with open(config["train_details"]["train_file"] , "wb") as f:
        pickle.dump(train_data , f , pickle.HIGHEST_PROTOCOL)
    with open(config["train_details"]["image_names"] , "wb") as f:
        pickle.dump(image_names , f , pickle.HIGHEST_PROTOCOL)

    print("training data and image_names are stored in piclke file.")
    print("******************************************************")

    val_data , val_images_names = read_files(config["preprocessing"]["val_data_filepath"] , config["preprocessing"]["val_datapath"])
    print(len(val_data))
    val_data = normalize_coords(val_data , config["preprocessing"]["val_datapath"])
    for index , img_path in enumerate(val_data):
        print(img_path)
        for object in val_data[img_path]:
            print(object)

    with open(config["train_details"]["test_path"] , "wb") as f:
        pickle.dump(val_data , f , pickle.HIGHEST_PROTOCOL)
    with open(config["train_details"]["val_image_names_path"] , "wb") as f:
        pickle.dump(val_images_names , f , pickle.HIGHEST_PROTOCOL)

    print("validation data and image_names are stored in piclke file.")
