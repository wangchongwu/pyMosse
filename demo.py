import cv2
import numpy as np
from glob import glob
import sys
import os
import time as t
#导入跟踪器
from Mosse import MosseTracker


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)           
            #frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            yield frame

def onControl(event, x, y, flags, param):
    img = param[0]
    traker = param[1]
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pass



def runtracker(video_path):

    #视频文件名
    if '/' in video_path:
        video_name = video_path.split('/')[-1].split('.')[0]
    elif '\\' in video_path:
        video_name = video_path.split('\\')[-1].split('.')[0]

    #创建跟踪器
    tracker = MosseTracker()
    #tracker = ScaledMosseTracker()


    #读一帧录像
    frame = next(get_frames(video_path))

    #创建录像路径
    if not os.path.exists('./demoVideo'):
        os.mkdir('./demoVideo')
    #初始化录像
    avifileName = './demoVideo/' + video_name + '_demo.avi'
    rate = 25
    video = cv2.VideoWriter(avifileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), rate,
                                    (frame.shape[1], frame.shape[0]))

    first_frame = True

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    #播放
    cnt = 0
    for frame in get_frames(video_path):
        cnt = cnt + 1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()

            #初始化跟踪器    
            tracker.init(frame, init_rect)
            first_frame = False
        else:

            #跟踪 
            outputs = tracker.track(frame)
            bbox = outputs

            #显示
            cv2.putText(frame, 'frame:%d' % cnt, (200,50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0),
                            1)
            cv2.putText(frame, 'resp:%f' % tracker.getMaxRespon(), (200,150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0),
                            1)
            

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 0, 255), 1)

            # cv2.putText(frame, 'time:%.1fms' % outputs['time'], (50,50), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (0, 0, 255),
            #                 1)

            # detBox = tracker.getDetectBBOX()
            # cv2.rectangle(frame, (detBox[0], detBox[1]),
            #               (detBox[0] + detBox[2], detBox[1] + detBox[3]),
            #               (255, 255, 255), 1)

            tup = (frame,tracker)
            cv2.setMouseCallback(video_name, onControl,tup)
            cv2.imshow(video_name, frame)
            video.write(frame)
            key = cv2.waitKey(20)
            
            if key & 0xFF == ord('q'):
                exit(0)



    # 关闭视频
    video.release()


if __name__ == '__main__':
    #path = R".\testData\car.avi"
    #path = R"D:\data_seq\VOT\vot2015\ball2\color"
    #path = R"D:\data_seq\20201102181152_001.raw.avi"
    #path = R"D:\data_seq\VOT\vot2018_lt\car6"
    #path = R"D:\data_seq\VOT\VOT2017and2018\wiper"
    #path = R"D:\AlgCompl\track_data\06 car2\car2.avi" 
    path = R"D:\data_seq\VOT\vot2018_lt\car9"
    #path = R"D:\AlgCompl\track_data\01 car\car.avi"
    #path = R"D:\data_seq\Dataset_UAV123\UAV123\data_seq\UAV123\wakeboard1"
    #path = R"D:\data_seq\VOT\\vot2014\\bicycle"
    runtracker(path)

