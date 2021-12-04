import time
import cv2
#from extractor import Extractor #does not do this makes circular import problems
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform




orb = cv2.ORB_create()

class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K=K
        self.Kinv =np.linalg.inv(self.K)

        
       

    def denormalize(self, pt):
        ret= np.dot(self.Kinv, [pt[0],pt[1], 1.0])
        print(ret)
        return int(round(ret[0])), int(round(ret[1]))
        #return int(round(pt[0]+self.w)), int(round(pt[1]+self.h))



    def extract(self, img): # in paint (extract is object functions are objects)
        #detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance=5) #numpy.AxisError: axis 3(must be 2 we said img=2) is out of bounds for array of dimension 3 quality level is (detected) dots
        #exctraction
        kps=[cv2.KeyPoint(x=f[0][0], y=f[0][0], _size=20) for f in feats] #key points 
        kps, des = self.orb.compute(img, kps)
       
        #matching
        #matches = []
        ret=[]
        if self.last is not None:
            matches= self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                   
                   kp1 = kps[m.queryIdx].pt
                   kp2 = self.last['kps'][m.trainIdx].pt
                   ret.append((kp1,kp2))
        
        

        #filter
        if len(ret) > 0:
            ret =np.array(ret)
           

            #normalized cords substract to move to 0
            ret[:, :, 0] == img.shape[0]/2
            ret[:, :, 1] == img.shape[1]/2
   
                
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    #EssentialMatrixTransform
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)


    
            ret = ret[inliers]

            s,v,d = np.linalg.svd(model.params)
            #print (v)
        #return
        self.last = {'kps':kps, 'des': des}                   
        return ret

