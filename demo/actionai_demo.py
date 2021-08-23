import os
import re
import sys
sys.path.append('.')
import cv2
import csv
import math
import time
import string
import random
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pprint import pprint
from collections import deque
from operator import itemgetter
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--video', help = 'video path',
                    default=int(0), type=str)
args = parser.parse_args()

body_labels = {0:'Nose', 1: 'Neck', 2: 'RShoulder', 3:'RElbow', 4:'RWrist', 5:'LShoulder', 6:'LElbow',
               7:'LWrist', 8:'RHip', 9:'RKnee', 10:'RAnkle', 11:'LHip', 12:'LKnee', 13:'LAnkle', 14:'REye',
              15:'LEye', 16:'REar', 17:'LEar'}
body_idx = dict([[v,k] for k,v in body_labels.items()])


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    '''
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    '''
    return ''.join(random.choice(chars) for x in range(size))

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (model_h, model_w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def inference(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    body_dict = draw_objects(image, counts, objects, peaks)
    return image, body_dict

def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def tracker_match(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_sum_assignment(-IOU_mat)
    matched_idx = np.asarray(matched_idx)
    matched_idx = np.transpose(matched_idx)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class PersonTracker(object):
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=10)
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict, frame_shape):
        '''
        ft_vec = np.zeros(2 * len(body_labels))
        for ky in pose_dict:
            idx = body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))
        self.q.append(ft_vec)
        ''' 
        # 기존의 ActionAI의 코드는 36개의 벡터로 표현하였는데 72개의 벡터로 수정
        self.dict = pose_dict
        ft_vec = np.zeros(2 * len(body_labels))
        full_vec = np.zeros(2 * len(body_labels))
        angle_vec = np.zeros(5)
        for ky in pose_dict:
            idx = body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))
            full_vec[2 * idx: 2 * (idx + 1)] = np.array(pose_dict[ky]) / np.array((frame_shape[1],frame_shape[0])) 
        
        vec = np.hstack([ft_vec, full_vec])
        self.q.append(vec)
        return
    def annotate(self, image):
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        image = cv2.putText(image, self.activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        image = cv2.drawMarker(image, self.centroid, (255, 0, 0), 0, 30, 4)
        return image

# update config file
update_config(cfg, args)   

model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

video_capture = cv2.VideoCapture(args.video)

w = int(video_capture.get(3))
h = int(video_capture.get(4))

fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
video_capture.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

DEBUG = False
WRITE2CSV = True
WRITE2VIDEO = True
RUNSECONDARY = False

if WRITE2CSV:
    activity = os.path.basename(args.video)
    dataFile = open('data/{}.csv'.format(activity),'w')
    newFileWriter = csv.writer(dataFile)

if WRITE2VIDEO:
    # Define the codec and create VideoWriter object
    name = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    write_video = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

if RUNSECONDARY:
    import tensorflow as tf
    secondary_model = tf.keras.models.load_model('models/lstm_spin_squat.h5')
    window = 3
    pose_vec_dim = 36
    motion_dict = {0: 'spin', 1: 'squat'}
trackers = []
if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(args.video)

    while True:
        bboxes = []
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        
        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')
                  
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
             
        out = draw_humans(oriImg, humans)
        for idx, body in enumerate(humans):
            bbox = get_bbox(list(body.tuple_list(oriImg)))
            bboxes.append((bbox, body.get_dictionary(oriImg)))
            

        track_boxes = [tracker.bbox for tracker in trackers]
        matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[0] for b in bboxes])
       
        
        for idx, jdx in matched:
            trackers[idx].set_bbox(bboxes[jdx][0])
            trackers[idx].update_pose(bboxes[jdx][1], out.shape)
    
        for idx in unmatched_detections:
            try:
                trackers.pop(idx)
            except:
                pass

        for idx in unmatched_trackers:
            person = PersonTracker()
            person.set_bbox(bboxes[idx][0])
            person.update_pose(bboxes[idx][1], out.shape)
            trackers.append(person)

        if RUNSECONDARY:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    sample = np.array(list(tracker.q)[:3])
                    sample = sample.reshape(1, pose_vec_dim, window)
                    pred_activity = motion_dict[np.argmax(secondary_model.predict(sample)[0])]
                    tracker.activity = pred_activity
                    image = tracker.annotate(image)
                    print(pred_activity)

        if DEBUG:
            pprint([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        if WRITE2CSV:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    newFileWriter.writerow([activity] + list(np.hstack(list(tracker.q)[:3])))

        if WRITE2VIDEO:
            write_video.write(out)
        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    try:
        dataFile.close()
    except:
        pass

    try:
        write_video.release()
    except:
        pass
