import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def get_iou(bb1, bb2):
    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou







im = np.array(Image.open('./annotation/20180625_1529925359865290_LF_139648.jpg'), dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(im)
boxA = None;
boxB = None;
with open('./annotation/20180625_1529925359865290_LF_139648_judgements.json') as json_file:
    data = json.load(json_file)
    for annotation in data:
        color = np.random.rand(3,)
        for obj in data[annotation]:
            for el in obj['boundaries']:
                if(obj["type"] == "Pedestrian"):
                    if(boxB == None):
                        if(boxA == None):
                            for coord in el['boundaryPoints']:
                                if coord['edge'] == 'Bottom':
                                    start_point = coord['coords']
                                if coord['edge'] == 'Top':
                                    top = coord['coords']
                                if coord['edge'] == 'Left':
                                    left = coord['coords']
                                if coord['edge'] == 'Right':
                                    right = coord['coords']
                            width = left[0] - right[0]
                            height = top[1] - start_point[1]
                            rect = patches.Rectangle((start_point[0], start_point[1]), width, height, linewidth=1,edgecolor=color, facecolor='none')
                            boxA = [start_point[0],start_point[1],width, height]
                            ax.add_patch(rect)
                        else:
                            for coord in el['boundaryPoints']:
                                if coord['edge'] == 'Bottom':
                                    start_point = coord['coords']
                                if coord['edge'] == 'Top':
                                    top = coord['coords']
                                if coord['edge'] == 'Left':
                                    left = coord['coords']
                                if coord['edge'] == 'Right':
                                    right = coord['coords']
                            width = left[0] - right[0]
                            height = top[1] - start_point[1]
                            rect = patches.Rectangle((start_point[0], start_point[1]), width, height, linewidth=1,edgecolor=color, facecolor='none')
                            boxB = [start_point[0],start_point[1],width, height]
                            ax.add_patch(rect)
                            print("IOU IS: ")
                            print(get_iou(boxA,boxB))



im = np.array(Image.open('./annotation/20180625_1529925359865290_RB_139648.jpg'), dtype=np.uint8)
fig, ax = plt.subplots(1)
ax.imshow(im)
with open('./annotation/20180625_1529925359865290_RB_139648_judgements.json') as json_file:
    data = json.load(json_file)
    for annotation in data:
        color = np.random.rand(3, )
        for obj in data[annotation]:
            for el in obj['boundaries']:
                for coord in el['boundaryPoints']:

                    if coord['edge'] == 'Bottom':
                        start_point = coord['coords']
                    if coord['edge'] == 'Top':
                        top = coord['coords']
                    if coord['edge'] == 'Left':
                        left = coord['coords']
                    if coord['edge'] == 'Right':
                        right = coord['coords']

                width = left[0] - right[0]
                height = top[1] - start_point[1]
                rect = patches.Rectangle((start_point[0], start_point[1]), width, height, linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
plt.show()
