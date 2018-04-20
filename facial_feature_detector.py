# -*-coding:utf-8-*
__author__ = 'Douglas and Iacopo'

import dlib
import os
import numpy as np

""" @brief 从保存所有特征点的dlib.full_object_detection对象中获取每一个特征点的横纵坐标, 生成包含68个特征点坐标的序列
    :param shape 保存所有特征点的dlib.full_object_detection对象
    :return 包含68个特征点的序列, 以元组的形式存储每一个特征点的横纵坐标：(x, y)
"""
def _shape_to_np(shape):
    xy = []
    for i in range(68):
        # shape.part(i)是第i个特征点, x和y属性分别对应该特征点的横纵坐标
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy

""" @brief 获取68个特征点标记序列
    :param img 源图像
    :param this_path 项目根目录
    :param image_path （个人添加）图像文件的相对路径
    :param write2File （个人添加）是否新建一个.pts数据文件, 记录68个特征点标记
    :return lmarks 返回保存所有人脸的特征点坐标的序列
"""
def get_landmarks(img, this_path, image_path, write2File = False):
    # if not automatically downloaded, get it from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    print this_path
    predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
    print predictor_path
    # dlib训练好的人脸检测器
    detector = dlib.get_frontal_face_detector()
    # dlib训练好的人脸特征检测器
    predictor = dlib.shape_predictor(predictor_path)

    lmarks = []
    # dets保存着所有人脸检测矩形的左上和右下坐标
    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    shapes = []
    # det保存着单个人脸检测矩形的左上和右下坐标
    for k, det in enumerate(dets):
        # 特征点全部保存在shape里面
        shape = predictor(img, det)
        shapes.append(shape)
        xy = _shape_to_np(shape)
        lmarks.append(xy)

    """个人添加部分："""
    ## 调用display_landmarks()方法
    display_landmarks(img, dets, shapes)
    ## 新建一个.pts数据文件, 记录68个特征点标记
    if write2File:
        pts_path = '\\' + image_path.split('.')[0] + '.pts'
        write_landmarks_to_file(this_path + pts_path, lmarks)

    lmarks = np.asarray(lmarks, dtype='float32')
    return lmarks

""" @brief 显示68特征点检测的效果图（原程序未调用该方法）
    :param img 源图像
    :param dets 保存着所有人脸检测矩形的左上和右下坐标
    :param shapes 保存所有检测到的人脸的特征点序列
"""
def display_landmarks(img, dets, shapes):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    for shape in shapes:
        win.add_overlay(shape)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

""" @brief （个人添加）新建一个.pts数据文件, 记录68个特征点标记
    :param pts_path 欲生成的.pts文件的相对路径
    :param lmarks 保存所有人脸的特征点坐标的序列
"""
def write_landmarks_to_file(pts_path, lmarks):
    f = open(pts_path, 'w', 0)
    f.write('# landmarks need to be in the form:\n')
    f.write('# x    y\n')
    for lmark in lmarks:
        print lmark
        for i in range(68):
            lmark_x = '%.3f' % lmark[i, 0]
            lmark_y = '%.3f' % lmark[i, 1]
            f.write(lmark_x + ' ' + lmark_y + '\n')
    f.close()