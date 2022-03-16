# 实现Mosse by wcw
# tips：1.使用固定模板大小，便于控制计算量和耗时
#       2.多尺度穷举，效果不佳



import numpy as np
import cv2



## tools 
# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

# pre-processing the image...
def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def random_warp(img):
    a = -45 / 16
    b = 45 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot



class MosseTracker:
    def __init__(self) -> None:
        self.Ai = None
        self.Bi = None
        self.G = None
        self.pos = None #当前帧目标框（x,y,w,h）
        self.clip_pos = None #当前帧搜搜区域，mosse与目标框一致
        
        self.lr = 0.125 #学习率
        self.sigma = 15
        
        self.rotate = False #使用随机投影样本
        self.num_pretrain = 8 #随机投影样本个数
        
        #固定模板
        self.scale = 1.0  #当前比例（相对于模板大小）
        #self.scaleFilter = [0.95,0.96,0.97,0.98,0.99,1,1.01,1.02,1.03,1.04,1.05] #尺度跟踪的尺度搜索范围
        self.scaleFilter = [0.95,1.0,1.05]
        
        self.TemplateSize = 128
        self.tempWidth = 0
        self.tempHeight = 0
        
        self.maxResp = 0.0
    
    def init(self,frame,bbox) -> bool:
        # 处理灰度图像
        if len(frame.shape)==3:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32)
            
        #生成当前波门 
        self.pos = list(bbox)    
        self.clip_pos = np.array([self.pos[0], self.pos[1], self.pos[0]+self.pos[2], self.pos[1]+self.pos[3]]).astype(np.int64)    
        # 获取初始化样本    
        fi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        # 高斯响应图生成
        response_map = self._get_gauss_response(frame, bbox)
        g = response_map[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        
        # 放缩到模板大小
        self.scale = max(self.pos[2],self.pos[3]) / self.TemplateSize
        self.tempWidth = int(self.pos[2] * self.scale)
        self.tempHeight = int(self.pos[3] * self.scale)
        fi = cv2.resize(fi,(self.tempWidth,self.tempHeight))
        g = cv2.resize(g,(self.tempWidth,self.tempHeight))
                
        #训练
        self.G = np.fft.fft2(g) #F(g)
        self.training(fi, self.lr)

    
    def track(self,frame) -> tuple:
        
        # 处理灰度图像
        if len(frame.shape)==3:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32)

        #多尺度检测
        response = []
        for _scale in self.scaleFilter:
            (x,y,w,h,r) = self.detect(frame,_scale)
            response.append(((x,y,w,h,r,_scale)))
            
        
        # 对响应排序
        response.sort(key = lambda x: -x[4])
        (x,y,w,h,r,_scale) = response[0]
        
        
        self.maxResp = r
        
        self.scale  *= _scale
        
        # 更新波门
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = w
        self.pos[3] = h

        
        # trying to get the clipped position [xmin, ymin, xmax, ymax]
        self.clip_pos[0] = np.clip(self.pos[0], 0, frame.shape[1])
        self.clip_pos[1] = np.clip(self.pos[1], 0, frame.shape[0])
        self.clip_pos[2] = np.clip(self.pos[0]+self.pos[2], 0, frame.shape[1])
        self.clip_pos[3] = np.clip(self.pos[1]+self.pos[3], 0, frame.shape[0])
        self.clip_pos = self.clip_pos.astype(np.int64)       
        

        
                

        # 模型更新'
        self.updateModel(frame)

                
        return (self.pos[0],self.pos[1],self.pos[2],self.pos[3])
    
    # 使用目标区域图像训练模型
    def training(self, init_frame, lr):
        G = self.G
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))

        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi)) * lr
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) * lr
        self.Ai = Ai
        self.Bi = Bi
        return Ai, Bi
    
    
    def detect(self,frame,_scale):
        
        #根据上一帧波门和当前检测尺度抠图  
        cx = self.pos[0] + 0.5 * self.pos[2]
        cy = self.pos[1] + 0.5 * self.pos[3]
        w = self.pos[2] * _scale
        h = self.pos[3] * _scale
        fi = frame[int(cy - h/2):int(cy + h/2),int(cx-w/2):int(cx+w/2)]
          
        #fi = frame[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        #重采样到模板大小
        fi = pre_process(cv2.resize(fi, (self.tempWidth, self.tempHeight)))
        
        #当前尺度包括尺度跟踪的尺度穷举以及放缩模板的尺度
        Allscale = self.scale/ _scale
        
        
    #    #计算响应图
        Hi = self.Ai / self.Bi
        Gi = Hi * np.fft.fft2(fi)
        
        gi = np.fft.ifft2(Gi)
        max_resp = np.max(gi)
        
        #gi = linear_mapping(gi)
        
        # 响应图极值点为当前帧目标位置
        max_pos = np.where(gi == max_resp)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)        
        
        # 更新波门
        x_s = int(self.pos[0] + dx * Allscale)
        y_s = int(self.pos[1] + dy * Allscale)
        w_s = int(self.pos[2] * _scale)
        h_s = int(self.pos[3] * _scale)
        

        return (x_s,y_s,w_s,h_s,max_resp)
        
    def updateModel(self,frame):
        #模型更新(跟踪阶段)
        fi = frame[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (self.tempWidth, self.tempHeight)))
        self.Ai = self.lr * (self.G * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * self.Ai
        self.Bi = self.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * self.Bi
  
    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)


        # cv2.imshow("",response * 255);
        # cv2.waitKey(0)
        return response
    
    def getDetectBBOX(self):
        dtbbox = (self.clip_pos[0],self.clip_pos[1],self.clip_pos[2]-self.clip_pos[0],self.clip_pos[3]-self.clip_pos[1])
        return dtbbox
    
    def getMaxRespon(self):
        return self.maxResp