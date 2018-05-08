# -*-coding:utf-8-*
__author__ = 'Iacopo'

import scipy.io as scio
import sklearn.metrics
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import ThreeD_Model
import camera_calibration as calib

# 如果我们翻转图像，重映射特征点的索引
## Index to remap landmarks in case we flip an image
repLand = [ 17,16,15,14,13,12,11,10, 9,8,7,6,5,4,3,2,1,27,26,25, \
            24,23,22,21,20,19,18,28,29,30,31,36,35,34,33,32,46,45,44,43, \
            48,47,40,39,38,37,42,41,55,54,53,52,51,50,49,60,59,58,57,56, \
            65,64,63,62,61,68,67,66 ]

""" @brief 如果当前路径下找不到output目录，则新建一个 
	:param output 输出图像目录的字符串值
"""
def mymkdir(output):
	if not os.path.exists(output):
		os.makedirs(output)

""" @brief 读取命令行参数，返回输入图像文件信息序列和输出文件目录
    :param argv 系统命令行参数序列 sys.argv
    :return fileList 输入图像文件信息序列，记录了输入图像文件所在目录、文件名、用到的特征点提取数据模型文件名
    return outputFolder 输出文件目录，默认/output/
"""
def parse(argv):
	fileList = []
	outputFolder = 'output/'
	## Case in which only an image is provided

	if len(argv) == 2:
        # 处理 python demo.py input/input_1.jpg
	    head, tail = os.path.split(argv[1])
	    fileList = [tail.split('.')[0]+','+str(argv[1])+',None']
	## Ok landmarks are provided as well or we are in batch mode
	elif len(argv) == 3:
	    #print argv[1]
	    ## If we are not in batch mode
	    if "--batch" not in str(argv[1]):
            # 处理 python demo.py input/input_1.jpg input/input_1.pts
	        head, tail = os.path.split(argv[1])
	        fileList = [tail.split('.')[0]+','+str(argv[1])+','+str(argv[2])]
	    else:
            # 处理 python demo.py --batch input/input.list
	        print '> Batch mode detected - reading from file: ' + str(argv[2])
	        filep = str(argv[2])
	        fileList = [line.strip() for line in open(filep)]
	else:
		print 'Usage for face rendering. See below'
		print 'Usage: python demo.py <image-path>'
		print 'Usage: python demo.py <image-path> <landmark-path>'
		print 'Usage: python demo.py --batch <file-list-path>'
		print 'where <file-list-path> is a csv file where each line has'
		print 'image_key,<image-path>,<landmark-path> (lines that contain # are skipped)'
		exit(1)
	return fileList, outputFolder

""" @brief 检测头部模型的姿态是否为正面
    :param pose 描述姿态的字符串值
    :return true表示模型姿态为正面，false则不为正面
"""
def isFrontal(pose):
	if '_-00_' in pose:
		return True
	return False

""" @brief 预加载.mat数据文件
    :param this_path 绝对路径
    :param pose_model_folder 头部模型数据文件所在的文件夹名
    :param pose_models 姿态数组
    :param nSub 每种3d头部模型的总个数
"""
def preload(this_path, pose_models_folder, pose_models,nSub):
    print '> Preloading all the models for efficiency'
    allModels= dict()
    for posee in pose_models:
        ## Looping over the subjects
        for subj in range(1,nSub+1):
            pose =   posee + '_' + str(subj).zfill(2) +'.mat'
            # load detections performed by dlib library on 3D model and Reference Image
            print "> Loading pose model in " + pose
            #model3D = ThreeD_Model.FaceModel(this_path + "/models3d_new/" + pose, 'model3D')
            if '-00' in posee:
                    model3D = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', True)
            else:
                    model3D = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', False)

			# pose: ThreeD_Model.FaceModel
            allModels[pose] = model3D
    return allModels

""" @brief 裁剪已绘制的输出视图
	:param pose
	:param frontal_raw 已绘制的输出视图
	:param crop_model 裁剪序列, 记录行和列的起始裁剪位置和终点裁剪位置
"""
def cropFunc(pose,frontal_raw,crop_model):
	frontal_raw = crop_face(frontal_raw, crop_model)
	return frontal_raw

""" @brief 裁剪已绘制的输出视图, 由cropFunc()调用
	:param img 已绘制的输出视图
	:param cropping 裁剪序列, 记录行和列的起始裁剪位置和终点裁剪位置
"""
def crop_face(img, cropping):
    if cropping is not None:
        img = img[cropping[1]:cropping[3],\
               cropping[0]:cropping[2],:]
        print '> Cropping with: ', cropping
    else:
        print '> No Cropping'
    return img

""" @brief 若输入图像（img）头部绕垂直轴旋转（摇摆）的角度（yaw）为负, 则更新68个特征点集合（lmarks）, 并水平翻转输入图像
    :param img 输入图像
    :param lmarks lmarks[0]保存着一个人脸68个特征点坐标列表
    :param allModels 存储所有FaceModel对象的字典
    :return img 若 yaw > 0, 返回原来的图像; 若 yaw < 0, 返回水平翻转后的图像
            lmarks 修改后的68个特征点的新集合
            yaw 绕垂直轴旋转（摇摆）的角度
"""
def flipInCase(img, lmarks, allModels):
	## Check if we need to flip the image
	# yaw（摇摆）表示头部绕垂直轴旋转
	yaws= []#np.zeros(1,len(allModels))
	## Getting yaw estimate over poses and subjects
    # 迭代每个FaceModel对象
	for mmm in allModels.itervalues():
 		proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(mmm, lmarks[0])
		yaws.append( calib.get_yaw(rmat) )
	yaws=np.asarray(yaws)
    # 计算 yaws 所有项的平均值
	yaw = yaws.mean()
	print '> Yaw value mean: ',  yaw
	if yaw  < 0:
	    print '> Positive yaw detected, flipping the image'
        # 水平翻转图像
	    img = cv2.flip(img,1)
	    # Flipping X values for landmarks
	    lmarks[0][:,0] = img.shape[1] - lmarks[0][:,0]
	    # Creating flipped landmarks with new indexing 
	    lmarks3 =  np.zeros((1,68,2))
	    for i in range(len(repLand)):
	        lmarks3[0][i,:] = lmarks[0][repLand[i]-1,:]
	    lmarks = lmarks3
	return img, lmarks, yaw

def show(img_display, img, lmarks, frontal_raw, \
	      face_proj, background_proj, temp_proj2_out_2, sym_weight):
    plt.ion()
    plt.show()
    plt.subplot(221)
    plt.title('Query Image')
    plt.imshow(img_display[:, :, ::-1])
    plt.axis('off')

    plt.subplot(222)
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1],c='red', marker='.',s=100,alpha=0.5)
    plt.axis('off')
    plt.subplot(223)
    plt.title('Rendering')

    plt.imshow(frontal_raw[:, :, ::-1])
    plt.axis('off')

    plt.subplot(224)
    if sym_weight is None:
	    plt.title('Face Mesh Projected')
	    plt.imshow(img[:, :, ::-1])
	    plt.axis('off')
	    face_proj = np.transpose(face_proj)
	    plt.plot( face_proj[1:-1:100,0], face_proj[1:-1:100,1] ,'b.')
	    background_proj = np.transpose(background_proj)
	    temp_proj2_out_2 = temp_proj2_out_2.T
	    plt.plot( background_proj[1:-1:100,0], background_proj[1:-1:100,1] ,'r.')
	    plt.plot( temp_proj2_out_2[1:-1:100,0], temp_proj2_out_2[1:-1:100,1] ,'m.')
    else:
	    plt.title('Face Symmetry')
	    plt.imshow(sym_weight)
	    plt.axis('off')
	    plt.colorbar()

    plt.draw()
    plt.pause(0.001)
    enter = raw_input("Press [enter] to continue.")
    plt.clf()

# def decidePose(yaw,opts):
# 	if opts.getboolean('renderer', 'nearView'):
# 		yaw = abs(yaw)
# 		# If yaw is near-frontal we render everything
# 		if yaw < 15:
# 			return [0,1,2]
# 		# otherwise we render only 2 profiles (from profile to frontal is noisy)
# 		else:
# 			return [1,2]
# 	else:
# 		return [0,1,2]

""" @brief 通过yaw的大小判断要绘制多少种姿态
	:param yaw 平均的头部模型摇摆角度（绕垂直轴旋转）
	:param opts 配置文件对象
	:param newModels 默认为True, 表示使用models3d_new文件夹中的头部模型
	:return 返回一个序列, 如要绘制 n 种姿态, 则该序列为 range(n)
"""
def decidePose(yaw,opts, newModels=True):
	if newModels == True:
	    if opts.getboolean('renderer', 'nearView'):
	        yaw = abs(yaw)
	        # If yaw is near-frontal we render everything
	        if yaw < 15:
	                return [0,1,2,3,4]
	        # otherwise we render only 2 profiles (from profile to frontal is noisy)                                                                                                                                                 
	        elif yaw > 40:
	                return [3,4]
	        else:
	                return [2,3,4]
	    else:
	        return [0,1,2,3,4]
	else:
		if opts.getboolean('renderer', 'nearView'):
			yaw = abs(yaw)
			# If yaw is near-frontal we render everything
			if yaw < 15:
				return [0,1,2]
			# otherwise we render only 2 profiles (from profile to frontal is noisy)
			else:
				return [1,2]
		else:
			return [0,1,2]
def decideSide_from_db(img, pose_Rt, allModels):
	## Check if we need to flip the image                                                                                                                                                                                            
	#model3D = ThreeD_Model.FaceModel(this_path + "/models3d/" + pose_models[0] +'_01.mat', 'model3D')                                                                                                                               
	## Getting yaw estimate over poses and subjects                                                                                                                                                                                  
	mm = allModels.values()[0]
	proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(mm, pose_Rt, pose_db_on=True)
	yaw =  calib.get_yaw(rmat)
	print '> Yaw value mean: ',  yaw
	return yaw