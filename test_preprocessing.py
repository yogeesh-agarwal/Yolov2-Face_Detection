import cv2
import sys
import math
import json
import numpy as np
import pre_processing as pre_process

def test_preprocess(config):
    batch_size = 10
    num_batch = 2
    anchors = np.array(config["model"]["anchors"])
    input_size = 416
    grid_size = 13
    anchors = anchors * input_size / (input_size / grid_size)
    batch_generator = pre_process.DataGenerator(416 ,
                                                13 ,
                                                anchors ,
                                                config["train_details"]["train_data_path"],
                                                config["train_details"]["train_file"] ,
                                                config["train_details"]["image_names"] ,
                                                config["model"]["classes"],
                                                False , True)
    print(batch_generator.num_train_instances)
    for index in range(num_batch):
        batch_images , batch_labels , best_anchor_indexes = batch_generator.load_data(batch_size , index)
        print("************************")
        for i in range(len(batch_images)):
            print("#########################")
            org_image = batch_images[i]
            label = batch_labels[i]
            object_index = 0
            for h in range(13):
                for w in range(13):
                    for a in range(5):
                        if label[h , w , a , 4] == 1:
                            try:
                                print(f"org_label : {label[h , w , a]}")
                                x_center = int(math.floor(label[h , w , a , 0] * 32))
                                y_center = int(math.floor(label[h , w , a , 1] * 32))
                                width = int(math.floor(label[h , w , a , 2] * 32))
                                height = int(math.floor(label[h , w , a , 3] * 32))
                                x1 = x_center - int(width/2)
                                y1 = y_center - int(height/2)
                                x2 = x1 + width
                                y2 = y1 + height
                                classes = "FACE" if label[h , w , a , 5] else "NON_FACE"
                                print(classes)
                                cv2.rectangle(org_image , (x1 , y1) , (x2 , y2) , (255 , 0  ,0) , 1)
                                print("detector_index : " , best_anchor_indexes[i][object_index] , "label : " , [x_center , y_center , width , height])
                                object_index += 1
                            except:
                                print(i , object_index , best_anchor_indexes , [x_center , y_center , width  , height])
                                sys.exit(0)
            cv2.imshow("images" , org_image[ : , : , ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    test_preprocess(config)
