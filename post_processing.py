import numpy as np
import cv2
import os
import pickle
import json

config_path = "./config.json"
with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

img_shape = [config["model"]["input_shape"] , config["model"]["input_shape"]]
nms_threshold = config["inference_details"]["nms_threshold"]
validation_filter_threshold = config["inference_details"]["validation_filter_threshold"]
camera_filter_threshold = config["inference_details"]["camera_filter_threshold"]
write_dir = config["inference_details"]["det_write_dir"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x , axis = -1 , t = -100):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x/e_x.sum(axis , keepdims = True)

def gen_cell_grid(grid_shape , num_anchors):
    x = np.arange(grid_shape[1])
    y = np.arange(grid_shape[0])
    x_cell , y_cell = np.meshgrid(x,y)
    x_cell = x_cell.reshape(-1,1)
    y_cell = y_cell.reshape(-1,1)
    cell_grid = np.concatenate([x_cell , y_cell], axis = -1)
    cell_grid = np.tile(cell_grid , [1 , num_anchors])
    cell_grid = np.reshape(cell_grid , [1, -1, 2])
    return cell_grid

def gen_anchor_grid(anchors , grid_shape):
    anchor_grid = np.tile(anchors , [grid_shape[0]*grid_shape[1] , 1])
    return anchor_grid

def modify_predictions(predictions , anchors):
    k = 4
    grid_shape = predictions.shape[1:3]
    num_anchors = predictions.shape[3]
    num_classes = predictions.shape[4] - 5
    strides = [img_shape[0] // grid_shape[0] , img_shape[1] // grid_shape[1]]
    predictions = np.reshape(predictions , [-1 , grid_shape[0]*grid_shape[1]*num_anchors , 5+num_classes])
    split_preds = np.split(predictions , [2,4,5] , axis = -1)
    box_centers , box_shapes , conf , classes = split_preds[0] , split_preds[1] , split_preds[2] , split_preds[3]
    cell_grid = gen_cell_grid(grid_shape , num_anchors)
    anchor_grid = gen_anchor_grid(anchors , grid_shape)

    box_centers = sigmoid(box_centers)
    box_centers = (box_centers + cell_grid) * strides
    box_shapes = sigmoid(box_shapes) * (anchor_grid).astype(np.float32) * strides * k
    confidence = sigmoid(conf)
    classes = softmax(classes)
    modified_predictions = np.concatenate([box_centers , box_shapes , confidence , classes] , axis = -1) #shape = [1,507,6]
    return modified_predictions

def transform_bbox(bbox):
    box_split = np.split(bbox , [1,2,3]  , axis = -1)
    x_center , y_center , w , h = box_split[0] , box_split[1] , box_split[2] , box_split[3]
    xmin = x_center - w/2
    ymin = y_center - h/2
    xmax = x_center + w/2
    ymax = y_center + h/2

    xmin = np.minimum(np.maximum(0.0 , xmin) , img_shape[0] - 1.0)
    ymin = np.minimum(np.maximum(0.0 , ymin) , img_shape[1] - 1.0)
    xmax = np.maximum(np.minimum(img_shape[0] - 1.0 , xmax) , 0.0)
    ymax = np.maximum(np.minimum(img_shape[1] - 1.0 , ymax) , 0.0)

    adjusted_w = xmax - xmin + 1.0
    adjusted_h = ymax - ymin + 1.0
    adjusted_x_center = xmin + (0.5 * adjusted_w)
    adjusted_y_center = ymin + (0.5 * adjusted_h)
    return np.concatenate([adjusted_x_center , adjusted_y_center , adjusted_w , adjusted_h] , axis = -1)

def iou(boxes , box):
    boxes_x_min = boxes[... , 0] - boxes[... , 2]*0.5
    boxes_y_min = boxes[... , 1] - boxes[... , 3]*0.5
    boxes_x_max = boxes[... , 0] + boxes[... , 2]*0.5
    boxes_y_max = boxes[... , 1] + boxes[... , 3]*0.5

    ref_x_min = box[0] - box[2]*0.5
    ref_y_min = box[1] - box[3]*0.5
    ref_x_max = box[0] + box[2]*0.5
    ref_y_max = box[1] + box[3]*0.5

    intersected_width = np.maximum(np.minimum(boxes_x_max , ref_x_max) - np.maximum(boxes_x_min , ref_x_min) , 0)
    intersected_height = np.maximum(np.minimum(boxes_y_max , ref_y_max) - np.maximum(boxes_y_min , ref_y_min) , 0)
    intersection = intersected_width * intersected_height
    union = (boxes[... , 2] * boxes[... , 3]) + (box[... , 2] * box[... , 3]) - intersection
    return intersection / union

def nms(locs , probs , threshold):
    pred_order = np.argsort(probs)[::-1]
    new_locs = locs[pred_order]
    new_porbs = probs[pred_order]
    keep = [True] * len(pred_order)

    for i in range(len(pred_order)-1):
        overlaps = iou(new_locs[i+1 : ] , new_locs[i])
        for j in range(len(overlaps)):
            if overlaps[j] > threshold:
                keep[pred_order[i+j+1]] = False
    return keep

def filter_predictions(localizations , det_probs , det_class , num_classes):
    max_prediction = 100
    if max_prediction < len(det_probs):
        pred_order = np.argsort(det_probs)[ : -max_prediction-1:-1]
        locs = localizations[pred_order]
        probs = det_probs[pred_order]
        cls_idx = det_class[pred_order]

        final_boxes = []
        final_probs = []
        final_class = []

        for c in range(num_classes):
            index_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
            keep = nms(locs[index_per_class] , probs[index_per_class] , nms_threshold)
            for i in range(len(keep)):
                if keep[i]:
                    final_boxes.append(locs[index_per_class[i]])
                    final_probs.append(probs[index_per_class[i]])
                    final_class.append(c)

        return [final_boxes , final_probs , final_class]

def draw_predictions(img , boxes , probs , classes , class_names , image_count , show_output , mode):
    global write_dir
    b_boxes = []
    for i in range(len(boxes)):
        xmin = int(boxes[i][0] - boxes[i][2]*0.5)
        ymin = int(boxes[i][1] - boxes[i][3]*0.5)
        xmax = int(boxes[i][0] + boxes[i][2]*0.5)
        ymax = int(boxes[i][1] + boxes[i][3]*0.5)
        conf = str(round(probs[i] , 2) * 100)
        pred_class = classes[i]
        label = class_names[pred_class]
        cv2.rectangle(img , (xmin , ymin) , (xmax , ymax) , (0 , 0 , 255) , 2)
        cv2.putText(img, label + " " + conf + "%" , (xmin , ymin) , cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 0), 2)
        b_boxes.append(["FACE" , probs[i] , xmin , ymin , int(boxes[i][2]) , int(boxes[i][3])])

    if show_output:
        if mode == "validation":
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            image_name = write_dir + "yolo_{}.jpg".format(image_count)
            cv2.imwrite(image_name, img)
            print(f"detection has been saved at  {write_dir}")

        cv2.imshow(mode+" testing" , img)
        if mode == "validation" or mode == "single":
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    return b_boxes

def post_process(images , predictions , anchors , pred_dict = None, image_names = None, show_output = True , eval_map = False , mode = "validation"):
    num_images = predictions.shape[0]
    predictions = modify_predictions(predictions , anchors)
    num_boxes = predictions.shape[1]
    num_classes = predictions.shape[2] - 5
    predictions = np.reshape(predictions , [-1 , num_boxes , 5 + num_classes]) # shape = [num_images , 845 , 25]
    split_preds = np.split(predictions , [4,5] , axis = -1)
    bbox_delta , confidence_scores , cls_pred = split_preds[0] , split_preds[1] , split_preds[2]
    adjusted_bbox_delta = transform_bbox(bbox_delta)
    combined_probs = np.multiply(cls_pred , confidence_scores)
    det_probs = np.amax(combined_probs , axis = -1)
    det_class = np.argmax(combined_probs , axis = -1)
    max_prob = np.amax(det_probs , axis = -1)

    if mode == "validation" or mode == "video" or mode == "single":
        filter_threshold = validation_filter_threshold
    elif mode == "camera":
        filter_threshold = camera_filter_threshold
    else:
        raise Exception("invalid mode of inference")

    for i,image in enumerate(images):
        final_boxes , final_probs , final_class = filter_predictions(adjusted_bbox_delta[i] , det_probs[i] , det_class[i] , num_classes)
        keep_index = [index for index in range(len(final_probs)) if final_probs[index] > filter_threshold]
        final_boxes = [final_boxes[index] for index in keep_index]
        final_probs = [final_probs[index] for index in keep_index]
        final_class = [final_class[index] for index in keep_index]
        boxes = draw_predictions(image , final_boxes , final_probs , final_class, ["FACE" , "NON_FACE"] , i , show_output , mode)
        if eval_map:
            image_name = image_names[i].split("/")[-1].split(".")[0]
            pred_dict.update({image_name : boxes})

    return pred_dict
