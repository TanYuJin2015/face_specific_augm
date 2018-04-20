# -*-coding:utf-8-*
import scipy.io as scio
import sklearn.metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})


class FaceModel:
    def __init__(self, path, name, getMask):
        self.load_model(path, name)
        self.eyemask = None
        # 正面时获取眼睛区域
        if getMask:
            # 获取眼睛区域, 特征点之间的连接线段宽度为8, 不用matplotlib显示最终结果
            self.eyemask = self.getEyeMask(width=8,plot=False)

    """ @brief 加载.mat数据文件, 并通过加载得到的数据设置 FaceModel 对象的各项属性
        :param path .mat数据文件的绝对路径
        :param name 'model3D'（.mat数据文件里的变量）
    """
    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3
        self.size_U = model['sizeU'][0, 0][0] #1x2
        self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3
        self.indbad = model['indbad'][0, 0]#0x1
        self.ref_U = np.asarray(model['refU'][0,0])
        self.facemask = np.asarray(model['facemask'][0,0])
        # self.facemask 每一项的值减1
        self.facemask-=1 #matlab indexing

    """ @brief 通过最小距离聚类算法得到脸部模型的眼睛特征点标记的位置序列, 并以此填充、绘制出眼睛区域
        :param width 线宽：填充眼睛区域的过程中, 相邻眼睛特征点之间互相要通过线段相连, width 表示的就是线段的宽度
        :param plot 是否用matplotlib显示眼睛区域的视图
        :return output 眼睛区域
    """
    def getEyeMask(self,width=1, plot=False):
        X = self.ref_U[:,:,0]
        # 将原来的X矩阵（224×224）转化为 1×50176 的向量
        X = X.reshape( (-1), order='F' )
        Y = self.ref_U[:,:,1]
        Y = Y.reshape( (-1), order='F' ) 
        Z = self.ref_U[:,:,2]
        Z = Z.reshape( (-1), order='F' )
        # 50176×3 : [X^T Y^T Z^T]
        cloud = np.vstack( (X,Y,Z) ).transpose()
        # 计算一个点与一组点之间的最小距离, 该函数为 X 中的每一行计算最接近的 Y 行的索引（根据指定的距离）, 最小距离也被返回
        # Y [idxs[i], ：] 是 Y 中与 X [i, ：] 最接近的行
        # 距离 dist 是 X 中第 i 行与 Y 中第 idxs [i] 行之间的距离
        [idxs, dist] = sklearn.metrics.pairwise_distances_argmin_min(self.model_TD, cloud)
        # 获取模型中左右眼特征点标记的位置序列
        eyeLeft = idxs[36:42]
        eyeRight = idxs[42:48]
        output1 = self.createMask(eyeLeft, width=width)
        output2 = self.createMask(eyeRight, width=width)
        output = output1 + output2
        # 是否用matplotlib显示眼睛区域(output)的视图
        if plot:
            plt.figure()
            plt.imshow(output)
            plt.draw()
            plt.pause(0.001)
            enter = raw_input("Press [enter] to continue.")

        output[output==255]=1
        return output

    """ @brief 眼睛区域填充函数：通过眼睛特征点标记的位置序列, 填充并绘制出眼睛区域 
        :param eyeLeft 眼睛特征点标记的位置序列
        :param width 线宽：填充眼睛区域的过程中, 相邻眼睛特征点之间互相要通过线段相连, width 表示的就是线段的宽度
        :return eyemaskfill 已填充的眼睛区域
    """
    def createMask(self,eyeLeft,width=1):
        eyeLefPix = np.unravel_index( eyeLeft, dims=self.ref_U.shape[::-1][1:3] ) 
        # 从位置坐标序列映射回行和列的序列
        # 如在224×224的图像中, eyeLeft[idx] = 17790, 则：
        # eyeLefPix[0, idx] = 17790 % 224 为 点eyeLeft[idx] 在图像中的行
        # eyeLefPix[1, idx] = 17790 / 224 为 点eyeLeft[idx] 在图像中的列
        eyeLefPix = np.asarray(eyeLefPix)
        eyemask = np.zeros((self.ref_U.shape[0]*self.ref_U.shape[1], 3))
        eyemask[eyeLeft,:] = 255
        eyemask = eyemask.reshape((self.ref_U.shape[0], self.ref_U.shape[1], 3), order='F')
        eyemask = eyemask.astype('uint8')
        eyemask = cv2.cvtColor(eyemask,cv2.COLOR_BGR2GRAY)

        # 相邻眼睛特征点之间互相通过线段相连
        for i in range(eyeLefPix.shape[1]):
            # cv2.line(点集, 起始点, 目标点, 颜色值, 线宽)
            cv2.line(eyemask,(eyeLefPix[0,i],eyeLefPix[1,i]),(eyeLefPix[0,(i+1)%eyeLefPix.shape[1]],\
                eyeLefPix[1,(i+1)%eyeLefPix.shape[1]]),(255,255,255),width)

        # 检测轮廓
        # _(返回的第1个变量): 显示源图像边缘的新图像
        # contours: 轮廓点序列
        # hierarchy: 轮廓的拓扑结构信息
        _, contours, hierarchy = cv2.findContours(eyemask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        eyemaskfill = np.zeros((self.ref_U.shape[0],self.ref_U.shape[1], 3))

        # 填充轮廓内的点
        # cv2.pointPolygonTest(某一轮廓序列, 像素点坐标, boolean)# 参数3: 若为 True, 则输出该像素点到轮廓的最近距离; 如果为False, 则输出为正表示在轮廓内, 0为轮廓上, 负为轮廓外。
        for r in range(self.ref_U.shape[0]):
            for c in range(self.ref_U.shape[1]):
                # 若像素点在轮廓内, 点亮该点
                if cv2.pointPolygonTest(contours[0], (c,r), False ) > 0:
                    eyemaskfill[r,c,:] = 255

        eyemaskfill = eyemaskfill.astype('uint8')

        return eyemaskfill