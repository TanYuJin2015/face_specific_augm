# -*-coding:utf-8-*
__author__ = 'Douglas and Iacopo'

import numpy as np
import cv2
import math

""" @brief 估计相机矩阵
    :param model3D 单个FaceModel对象
    :param fidu_XY 人脸68个特征点坐标列表
    :param pose_db_on 默认为False
    :return projection_matrix 相机矩阵 M（M = K [R|t]）
            model3D.out_A：固有摄像机内参数：K
            rmat, tvec：摄像机外参数, 旋转和平移矩阵：R 和 t
"""
def estimate_camera(model3D, fidu_XY, pose_db_on=False):
    if pose_db_on:
        rmat, tvec = calib_camera(model3D, fidu_XY, pose_db_on=True)
        tvec = tvec.reshape(3,1)
    else:
        # R^I和t^I矩阵
        rmat, tvec = calib_camera(model3D, fidu_XY)
    # 旋转和平移矩阵：[R|t]
    RT = np.hstack((rmat, tvec))
    # 相机矩阵：M = K [R|t]
    projection_matrix = model3D.out_A * RT
    return projection_matrix, model3D.out_A, rmat, tvec

""" @brief 校准（去掉表现不好的特征点）并计算旋转和平移矩阵：R 和 t（calibrate 校准）
    :param model3D 单个FaceModel对象
    :param fidu_XY 人脸68个特征点坐标列表
    :param pose_db_on 默认为False
    :return rmat, tvec：旋转和平移矩阵：R和t
"""
def calib_camera(model3D, fidu_XY, pose_db_on=False):
    #compute pose using refrence 3D points + query 2D point
    ## np.arange(68)+1 since matlab starts from 1
    # 使用参考3D点+查询2D点计算姿态
    if pose_db_on:
        rvecs = fidu_XY[0:3]
        tvec = fidu_XY[3:6]
    else:
        # 找出存在于第一个集合但不存在于第二个集合中的元素, 返回一个数组集合（注意：Matlab的起始索引为1）
        goodind = np.setdiff1d(np.arange(68)+1, model3D.indbad)
        # [0, 1, 2, ... , 67]
        goodind=goodind-1
        # 此步可以去掉特征点集中表现不好的特征点
        fidu_XY = fidu_XY[goodind,:]
        # 给定 物体3D点集model3D.model_TD 与对应的 图像2D点集, 以及摄像头内参数的情况下计算物体的3D姿态
        # rvec 是旋转关系向量, tvec 是平移关系向量
        ret, rvecs, tvec = cv2.solvePnP(model3D.model_TD, fidu_XY, model3D.out_A, None, None, None, False)
    # 使用Rodrigues变换将 rvecs 变换为旋转矩阵 rmat
    rmat, jacobian = cv2.Rodrigues(rvecs, None)

    inside = calc_inside(model3D.out_A, rmat, tvec, model3D.size_U[1], model3D.size_U[0], model3D.model_TD)
    # 若没有特征点在裁剪区域内, 则将模型绕z轴（在世界坐标系中）180°旋转, 使其移到裁剪区域内
    if(inside == 0):
        tvec = -tvec
        t = np.pi
        RRz180 = np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
        rmat = RRz180*rmat
    return rmat, tvec

""" @brief 计算出3D头部模型需绕 y 轴旋转以对齐二维特征点的角度（3D特征点和二维特征点的对齐,.即姿态的对齐）
    :param rmat 旋转矩阵：R
    :return out_yaw 3D头部模型需绕 y 轴（垂直轴）旋转以对齐二维特征点的角度
"""
def get_yaw(rmat):
    modelview = rmat
    modelview = np.zeros( (3,4 ))
    modelview[0:3,0:3] = rmat.transpose()
    modelview = modelview.reshape(12)
    # Code converted from function: getEulerFromRot()
    # 计算出3D头部模型需绕 y 轴旋转以对齐二维特征点的角度（3D特征点和二维特征点的对齐,.即姿态的对齐）
    angle_y = -math.asin( modelview[2] )  # Calculate Y-axis angle                                                                                                                                                                       
    C = math.cos( angle_y)
    angle_y = math.degrees(angle_y)

    if np.absolute(C) > 0.005: #  0
        trX = modelview[10] / C # No, so get X-axis angle                                                                                                                                                                                
        trY = -modelview[6] / C
        angle_x = math.degrees( math.atan2( trY, trX ) )

        trX = modelview[0] / C  # Get z-axis angle                                                                                                                                                                                       
        trY = - modelview[1] / C
        angle_z = math.degrees(  math.atan2( trY, trX) )
    else:
        # 正脸时, 俯仰被锁定
        # Gimball lock has occured                                                                                                                                                                                                       
        angle_x = 0
        trX = modelview[5]
        trY = modelview[4]
        angle_z = math.degrees(  math.atan2( trY, trX) )

    # Adjust to current mesh setting                                                                                                                                                                                                     
    angle_x = 180 - angle_x
    angle_y = angle_y
    angle_z = -angle_z

    out_pitch = angle_x
    out_yaw = angle_y
    out_roll = angle_z

    return out_yaw

""" @brief 计算摄像机的外参矩阵[R|t]和透视投影矩阵（用于将照相机坐标系转换为裁剪坐标系）
    :param camera_matrix 固有摄像机参数：K
    :param rmat 旋转矩阵：R
    :param tvec 平移矩阵：t
    :param width 输出视图宽度
    :param height 输出视图高度
    :return mv 摄像机的外参矩阵[R|t]
            projection_matrix 透视投影矩阵（针孔相机模型）
"""
def get_opengl_matrices(camera_matrix, rmat, tvec, width, height):
    projection_matrix = np.asmatrix(np.zeros((4,4)))
    near_plane = 0.0001
    far_plane = 10000

    # fx, fy表示焦距的参数, 控制2D缩放; px, py表示主点偏移, 控制2D平移
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    px = camera_matrix[0,2]
    py = camera_matrix[1,2]

    # 计算透视投影矩阵（针孔相机模型）
    projection_matrix[0, 0] = 2.0 * fx / width
    projection_matrix[1, 1] = 2.0 * fy / height
    projection_matrix[0, 2] = 2.0 * (px / width) - 1.0
    projection_matrix[1, 2] = 2.0 * (py / height) - 1.0
    projection_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    projection_matrix[3, 2] = -1
    projection_matrix[2, 3] = -2.0 * far_plane * near_plane / (far_plane - near_plane)

    deg = 180
    t = deg*np.pi/180.
    RRz=np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
    RRy=np.asmatrix([np.cos(t), 0, np.sin(t), 0, 1, 0, -np.sin(t), 0, np.cos(t)]).reshape((3, 3))
    rmat=RRz*RRy*rmat

    mv = np.asmatrix(np.zeros((4, 4)))
    mv[0:3, 0:3] = rmat
    mv[0, 3] = tvec[0]
    mv[1, 3] = -tvec[1]
    mv[2, 3] = -tvec[2]
    mv[3, 3] = 1
    return mv, projection_matrix

""" @brief 计算构成裁剪区域的6个平面方程
    :param camera_matrix 固有摄像机参数：K
    :param rmat 旋转矩阵：R
    :param tvec 平移矩阵：t
    :param width 输出视图宽度
    :param height 输出视图高度
    :return frustum 存储着构成裁剪区域的6个平面方程的矩阵
"""
def extract_frustum(camera_matrix, rmat, tvec, width, height):
    # 计算摄像机的外参矩阵[R|t]和透视投影矩阵
    mv, proj = get_opengl_matrices(camera_matrix, rmat, tvec, width, height)
    # 裁剪坐标系的变换矩阵
    clip = proj * mv
    frustum = np.asmatrix(np.zeros((6 ,4)))
    #/* Extract the numbers for the RIGHT plane */
    frustum[0, :] = clip[3, :] - clip[0, :]
    #/* Normalize the result */
    v = frustum[0, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[0, :] = frustum[0, :]/t

    #/* Extract the numbers for the LEFT plane */
    frustum[1, :] = clip[3, :] + clip[0, :]
    #/* Normalize the result */
    v = frustum[1, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[1, :] = frustum[1, :]/t

    #/* Extract the BOTTOM plane */
    frustum[2, :] = clip[3, :] + clip[1, :]
    #/* Normalize the result */
    v = frustum[2, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[2, :] = frustum[2, :]/t

    #/* Extract the TOP plane */
    frustum[3, :] = clip[3, :] - clip[1, :]
    #/* Normalize the result */
    v = frustum[3, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[3, :] = frustum[3, :]/t

    #/* Extract the FAR plane */
    frustum[4, :] = clip[3, :] - clip[2, :]
    #/* Normalize the result */
    v = frustum[4, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[4, :] = frustum[4, :]/t

    #/* Extract the NEAR plane */
    frustum[5, :] = clip[3, :] + clip[2, :]
    #/* Normalize the result */
    v = frustum[5, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[5, :] = frustum[5, :]/t
    return frustum

""" @brief 计算在裁剪区域内的特征点的数目
    :param camera_matrix 固有摄像机参数：K
    :param rmat 旋转矩阵：R
    :param tvec 平移矩阵：t
    :param width 输出视图宽度
    :param height 输出视图高度
    :param obj_points 三维头部模型特征点集合
    :return inside 在裁剪区域内的特征点的数目
"""
def calc_inside(camera_matrix, rmat, tvec, width, height, obj_points):
    # 计算构成裁剪区域的6个平面方程, 平面各系数均存在 frustum 矩阵中
    frustum = extract_frustum(camera_matrix, rmat, tvec, width, height)
    inside = 0
    for point in obj_points:
        # 判断三维头部特征点是否在裁剪区域内, 是则 inside + 1
        if(point_in_frustum(point[0], point[1], point[2], frustum) > 0):
            inside += 1
    return inside

""" @brief 判断点(x, y, z)是否在裁剪区域内
    :param x 点的x坐标
    :param y 点的y坐标
    :param z 点的z坐标
    :param frustum 裁剪面矩阵
    :return boolean 若返回True, 则点(x, y, z)在裁剪区域内，若False, 则点(x, y, z)不在裁剪区域内
"""
def point_in_frustum(x, y, z, frustum):
    for p in range(0, 3):
        if(frustum[p, 0] * x + frustum[p, 1] * y + frustum[p, 2] + z + frustum[p, 3] <= 0):
            return False
    return True