## Yolov2-Face_Detection
**Yolov2 model from scratch for Face detection**  

**Description**  
A face detection model which uses yolov2 structure to detect faces in a given image , video or input from webcam. Yolov2 official research paper is used as the reference for structure and loss computation for yolov2 including some modification in loss fucntion to stabilise the training process.

**Requirements**  
1. Python3.6+
2. Tensorflow1.14+
3. numpy
4. Cv2
5. matplotlib
6. imgaug

To replicate the exact environment as used while creating this project , please use the requirements.txt file to install all the required packages in a virtual environment using following command:  
```
sudo apt-get install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```  
Note : Please download the [wider face dataset](http://shuoyang1213.me/WIDERFACE/) and place it under data folder, to continue further as training and testing of this model requires this dataset.  


**Training**  
Before Starting the training , be ready with the pre-trained classification weights for FACE-Non_Face dataset. These are used as weights for the base network (darkent 19) , and actual training is for detection module. It is totally fine to start training from scratch , but after experimenting with both methods, i concluded that pre-trained weights tend make the training smooth and converge nicely. Not much difference w.r.t mAP was observed in the end.   
***For Classification model for face-Non_Face ,  [refer here](https://github.com/yogeesh-agarwal/Face-Classification-Model)***

Steps to train the model:  
1. Create a file for desired dataset for face detection. Default dataset used is wider face dataset which is considered to be best benchmark dataset for current sota face_detection and related tasks. Model expects two pickle files for training , one containing a dictionary with key as img_names and value as list of list of object or boxes dimension , and the other containing just the image_names. This makes the preprocessing part of the model little more optimised (still working on cpu) , allowing the preprocessing happening dynamically while the program is running.  
Currently wider face dataset is pruned to ignore the faces whose area intersection with original_image % is < 0.01.
The obvious reason to prune the dataset , because of the nature of the dataset , which contains pretty small boxes in good lot of images, and hence when resized to model input shape just becomes practically insensible to train model on such small faces.

2. Generate the anchors for the dataset and choose the anchors based on what is the cluster IOU.  
command : ```python build_anchors.py```

3. plot the anchors if needed to visualize how the anchors look on original image and thier shape.  
command : ```python plot_anchors.py```

4. Test the preprocessing to check for proper encoding of boxes and data_augmentation.  
command : ```python test_preprocessing.py```

5. Following above steps ,ensures that everything is ready for training. Start training and monitor the loss.  
command : ```python train.py```

Note : Current loss function used to train the model is very similar to the one used in the [official paper of Yolov2](https://arxiv.org/abs/1612.08242) , difference being in the way , coordinate loss is computed where instead of using EXP for wh_loss , sigmoid is used with the combination of a adjuster_parameter which governs how big an anchor can adjust to given its original size. [For more info](https://github.com/ultralytics/yolov3/issues/168).  

**Testing**  
Current version of code supports 4 modes of testing:  
1. testing using wider face validation set / testing set. Filter threshold and nms threshold are two parameters which can be tweked for comparing the detections and their nature.
command : ```python test.py ``` (default mode)  

2. Using web_cam to test using device's web camera.  
command : ```python test.py --mode camera```

3. providing an image path in command line for image to be tested.  
command : ```python test.py --mode single --img_path path_to_image```  


4. using video to test on face detection model.  
command : ```python test.py --mode video --video_path path_to_video```

**Evaluation**  
To evaluate the mAP for the current model , clone the [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) and place in the current directory.  
After specifying the number of teating images and starting index in config file , run the evaluation file to get the mAP for face class.  
command : ```python evaluate_mAP.py```  

**Results**  
some detection results :  
--mode = "**validation**" :  
<p align="center">
  <img width="300" height="300" src="https://github.com/yogeesh-agarwal/Yolov2-Face_Detection/blob/main/data/detections/yolo_0.jpg">
</p>
<p align="center">
  <img width="300" height="300" src="https://github.com/yogeesh-agarwal/Yolov2-Face_Detection/blob/main/data/detections/yolo_3.jpg">
</p>  
more detections in the detections folder.

Here is the list of mAP at different configuration :  

| filter_threshold  | NMS_threshold  | min IOU  | mAP     |
| :-------------:   |:-------------: |:-------: | :-----: |
| 0.5               | 0.4            |   0.4    |  56.04  |
| **0.2**           | **0.4**        | **0.4**  |**72.03**|
| 0.1               | 0.4            |   0.4    |  77.16  |
| 0.5               | 0.4            |   0.5    |  53.61  |
| **0.2**           | **0.4**        | **0.5**  |**66.31**|
| 0.1               | 0.4            |   0.5    |  69.84  |
