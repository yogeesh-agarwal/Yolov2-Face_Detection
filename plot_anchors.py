import numpy as np
import json
import cv2

def plot_anchors(anchors , height , width):
    image = np.zeros([height , width , 3])
    center_x = width // 2
    center_y = height // 2
    for anchor in anchors:
        x1 = int(center_x - (anchor[0] / 2))
        y1 = int(center_y - (anchor[1] / 2))
        x2 = int(center_x + (anchor[0] / 2))
        y2 = int(center_y + (anchor[1] / 2))
        height = anchor[1]
        width = anchor[0]
        anchor_area = width * height
        print(height , width , anchor_area)
        print("percentage area : " , (anchor_area / (416 * 416)) * 100)
        print(x1, y1, x2, y2 , center_x, center_y , anchor)
        cv2.rectangle(image , (x1,y1) , (x2,y2) , (255 , 0 , 0) , 1)

    cv2.rectangle(image , (1,1) , (416 , 416) , (0,0,255) , 1)
    cv2.imshow("image" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    anchors = np.array(config["model"]["anchors"])
    input_size = 416
    grid_size = 13
    anchors = np.multiply(anchors , 416)
    plot_anchors(anchors , input_size , input_size)
