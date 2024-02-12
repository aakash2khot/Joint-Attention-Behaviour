# python3 demo_new.py  --snapshot models/L2CSNet_gaze360.pkl  --file Videos/10sec_no_sound.mp4 --gpu -1

import argparse
import numpy as np
import cv2
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from my_utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

# from darknet.darknet import load_network

def score1(p1, dp1, p2, dp2):
    sx=dp2[0]*dp1[0]+dp2[1]*dp1[1]
    sy=dp2[0]*(-dp1[1])+dp2[1]*(dp1[0])
    if sy==0:
        return None
    PQx = p2[0] - p1[0]
    PQy = p2[1] - p1[1]
    rx = dp1[0]
    ry = dp1[1]
    rxt = -ry
    ryt = rx
    qx = PQx * rx + PQy * ry
    qy = PQx * rxt + PQy * ryt
    a = qx - qy * sx / sy
    return (p1[0]+a*rx, p1[1]+a*ry)

def score2(inter):
    if len(inter)==1:
        return 0
    if inter[-1]==None or inter[-2]==None:
        return 0
    dist=(inter[-2][0]-inter[-1][0])**2+(inter[-2][1]-inter[-1][1])**2
    if(dist<25):
        return 1
    return 0

    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera id to be processed',  
        default=0, type=int)
    parser.add_argument(
        '--file',dest='fname', help='Video file to be processed',  
        default='', type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def is_object_detected_around_point(point, detections):
    object_boxes = detections.xyxy[0].cpu().numpy()
    for box in object_boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        distance = math.sqrt((point[0] - x_center) ** 2 + (point[1] - y_center) ** 2)
        # print("Distance between obj and point: ",distance)
        # Distance and all are measured in pixels 
        if distance < 1000:
            return True
    return False

def gazeAtPerson(gaze, face):
    tmin = 0.0
    tmax = 10000000000000.0

    for d in range(2):
        t1 = (face[0][d] - gaze[0][d]) /gaze[1][d]
        t2 = (face[1][d] - gaze[0][d]) /gaze[1][d]

        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))

    return tmin < tmax


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    cam = args.cam_id
    video_file = args.fname
    # gpu = select_device(args.gpu_id, batch_size=batch_size)
    # new change
    gpu = select_device(str(args.gpu_id), batch_size=batch_size)

    snapshot_path = args.snapshot
   
    

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)
    # detector = RetinaFace(gpu_id=-1)
    # new change
    detector = RetinaFace(gpu_id=-1)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0
  
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    inter=[]
    jab_score=0
    jab_score2=0
    incr=1
    t=0
    print(fps)

    object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open file")
    
    frame_count = 0
    total_frame_count = 0

    with torch.no_grad():
        # yolo = YOLOv3("darknet/cfg/yolov3.cfg", "yolov3.weights")

        # result = cv2.VideoWriter('results.mp4', 
        #         fourcc,
        #         fps, (frame_width,frame_height))
        while cap.isOpened():
            success, frame = cap.read()    
            start_fps = time.time()  
            total_frame_count+=1

            if not success:
                break

            faces = detector(frame)
           

            p=[]
            d=[]
            f=[]
            if faces is None:
                print("Faces not detected in frame ", frame_count)
            if faces is not None: 
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                
                    f.append([(x_min,y_min),(x_max,y_max)])
                    _,dx,dy=draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                    p.append(((2*x_min+bbox_width)/2.0, (2*y_min+bbox_height)/2.0))
                    magn=math.sqrt(dx**2+dy**2)
                    dx=dx/magn
                    dy=dy/magn
                    d.append((dx,dy))

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                t=t+1
                if len(p)==1:
                    print("Only 1 face detected in frame ",total_frame_count)
                if len(p)==0:
                    print("No face there in ",total_frame_count)
                if len(p)>1:
                    frame_count+=1
                    t=0
                    point=score1(p[0],d[0],p[1],d[1])
                    inter.append(point)
                    if score2(inter)==1:
                        incr=incr+1
                    else:
                        incr=1
                    if point is not None:
                        jab_score=jab_score+incr
                        if is_object_detected_around_point(point, object_detector(frame)) or gazeAtPerson([p[0],d[0]],f[1]) or gazeAtPerson([p[1],d[1]],f[0]):
                           print("Object detected around the point or looking at each other")
                           jab_score2=jab_score2+incr
                        #    print(jab_score2)
                        else:
                            print("No object detected around the point")
                    
                    
                        
                    # print("Point is", point)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            # result.write(frame)
            #cv2.imshow("Demo",frame)
            # print("Processing frames")
            #if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty('Demo', cv2.WND_PROP_VISIBLE) < 1:
             #   break
#            success,frame = cap.read()

        # result.release()
        cap.release()
        cv2.destroyAllWindows()
        # print("The video was successfully saved")
        print("Frame count: ", total_frame_count)
        print("Frame count: ", frame_count)
        print("JAB score is ", jab_score)
        print("JAB score is (Normalized)", jab_score/frame_count)
        print("JAB score with Obj detection", jab_score2)
        print("JAB score with Obj detection (Normalized)", jab_score2/frame_count)
        
        # print("JAB score is", jab_score2)

