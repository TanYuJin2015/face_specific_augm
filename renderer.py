# -*-coding:utf-8-*
import scipy.io as scio
import sklearn.metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

class Params:
    ACC_CONST = 2e3#5e5 #<--- obsolete
    ksize_acc = 15#15
    ksize_weight = 33#33

""" @brief 获取可判断对应索引位置的点是否投影到输入图像外的一个序列
    :param project 二维坐标集, 2×50176, 第一行为x坐标, 第二行为y坐标
    :param img 输入图像
    :return bad 一个1×50176的序列, 该序列可判断对应索引位置的点是否投影到输入图像外, 序列中每个元素是一个布尔值, True表示投影到输入图像外
"""
def badIndex(project, img):
    # matlab index
    # 坐标中有小于1的值 OR y坐标大于224 OR x坐标大于224
    bad = np.logical_or(project.min(axis=0) < 1, project[1, :] > img.shape[0])
    bad = np.logical_or(bad, project[0, :] > img.shape[1])
    bad = np.asarray(bad).reshape((-1), order='F')
    return bad

""" @brief 绘制输出视图, 通过 prj 将输入图像的点重映射到输出视图的各个点（输出视图背景有被拉长的表现是因为投影点x坐标值的重叠）
    :param img 输入图像
    :param t_height 输出视图的高
    :param t_width 输出视图的宽
    :param prj 待重映射的50176(224×224)个像素点的坐标序列（从输入图像重映射到输出视图）
    :param idx 输出视图的所有像素点的索引序列
    :return new_img 输出视图
"""
def warpImg(img, t_height, t_width, prj, idx):
    new_img = np.zeros((t_height*t_width, 3))
    ## In case we have some points
    if prj.size != 0:
        # 重映射：即把一幅图像中某位置的像素放置到另一个图片指定位置的过程
        pixels = cv2.remap(img, np.squeeze( np.asarray( prj[0,:] ) ).astype('float32'),\
         np.squeeze( np.asarray( prj[1,:] ) ).astype('float32'),  cv2.INTER_CUBIC)
        # pixels_copy = pixels.reshape((224, 224, 3), order='F')
        # pixels_copy[pixels_copy > 255] = 255
        # pixels_copy[pixels_copy < 0] = 0
        # pixels_copy = pixels_copy.astype('uint8')
        # cv2.imshow("pixels_copy", pixels_copy)
        # cv2.waitKey()
        pixels = pixels[:,0,:]
        new_img[idx,:] = pixels
    else:
        print '> Projected points empty'
    new_img = new_img.reshape(( t_height, t_width, 3), order='F')
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    new_img = new_img.astype('uint8')
    return new_img

""" @brief 规范化所有投影到输入图像外的点（NDC过程）
    :param out_proj 投影到输入图像外的点集
    :return out_proj 经规范化后的点集
"""
def NormalizePoints(out_proj):
    # 在成像平面外的点集的所有坐标值中, 获取最大的x（相对于成像平面外的最右方）和最大的y（相对于成像平面外的最上方）
    maxOut = out_proj.max(axis=1)
    # 在成像平面外的点集的所有坐标值中, 获取最小的x（相对于成像平面外的最左方）和最小的y（相对于成像平面外的最下方）
    minOut = out_proj.min(axis=1)
    # 存储着若要所有映射的二维点都能在成像平面内, 成像平面的长和高需要的最小值
    lenn =maxOut-minOut
    ## Normalize the points somehow inside the image
    ## In theory here we can do better to avoid putting in the background pixel of the faces
    # 在图像内以某种方式规范化点
    # 从理论上讲，我们可以做得更好，避免将脸部当作背景像素来放入图像
    den = maxOut-minOut
    den[den==0]=1
    # 规范化运算
    out_proj = (out_proj-minOut)/den
    return out_proj, lenn

""" @brief 反规范化矩阵
    :param out_proj 经规范化过的矩阵
    :param size 
    :return 反规范化后的矩阵
"""
def UnnormalizePoints(out_proj, size):
    return np.multiply(out_proj,size.T)

""" @brief 当有一些投影点在输入图像外的右侧时, 处理背景（ 对应 左侧背景 + 论文Ⅳ.C中(d)图的第二个平面(经移动处理的右侧背景) ）
           该背景区域大小与这些点形成的区域大小相等, 背景区域以采样点 thWidth 为结束点（第二个平面最右端的点）
    :param out_proj 投影到输入图像外的点集
    :param face_proj_in 投影到输入图像内的脸部点集
    :param img 输入图像
    :param opts 配置文件对象
    :return out_proj 经背景处理后的点集
"""
def HandleBackground(out_proj,face_proj_in, img, opts):
    if out_proj.size != 0:
        out_proj,lenn = NormalizePoints(out_proj)
        widthX = lenn[1]
        heightY = lenn[0]
        # face_proj_in[0,:].min()/img.shape[1] 是二维图像上脸部最左端的点经规范化的x坐标的值
        # scaleFaceX使得对背景进行采样时, 使采样点稍微离脸部隔点距离
        thWidth = face_proj_in[0,:].min()/img.shape[1]*opts.getfloat('renderer','scaleFaceX')
        # idxOveral 为 所有经规范化的x坐标值大于 thWidth 的点的索引位置
        idxOveral =  np.nonzero(np.squeeze(np.asarray(out_proj[0,:]))>thWidth)[0]
        if idxOveral.size != 0:
            # 将右侧的这些点作为背景（从人脸的左侧位置插入, 对应论文Ⅳ.C中(d)图的第二块平面）移动到采样点
            out_proj[0,idxOveral] = out_proj[0,idxOveral]/out_proj[0,idxOveral].max()*thWidth

        # In case we want to skip the head and go in the right part of the face
        # diffX = out_proj[0,idxOveral]-thWidth#=thWidth
        # #print diffX
        # rempPts = thWidthMax + diffX[0,:]
        # rempPts, lenn = NormalizePoints(rempPts)
        # rempPts = face_proj_in[0,:].max()*1.1 + UnnormalizePoints(rempPts, img.shape[0]-face_proj_in[0,:].max()*1.1 )
        out_proj = UnnormalizePoints(out_proj, np.matrix([img.shape[1],img.shape[0] ]) )
    return out_proj

""" @brief 绘制输出视图的完整过程的方法
    :param img 输入图像（可能经过水平翻转处理）
    :param proj_matrix 相机矩阵 M（M = K [R|t]）
    :param ref_U 三维头部模型的坐标集（包含脸部和背景区域等所有点的坐标）
    :param eyemask 眼睛区域的点的索引集
    :param facemask 脸部区域的点的索引集
    :param opts 配置文件对象
    :return frontal_raw 绘制的输出图像
    :return frontal_sym 应用了soft-sym处理的输出图像
    :return face_proj_in 投影到输入图像内的脸部区域的点集
    :return out_proj_disp 投影到输入图像外的点集
    :return out_proj 投影到输入图像外的点集
    :return sym_weight
"""
def render(img, proj_matrix, ref_U, eyemask, facemask, opts):
    print "> Query image shape:", img.shape
    img = img.astype('float32')
    ### Projecting 3D model onto the the image
    # threedee.shape == (3L, 50176L)

    threedee = np.reshape(ref_U, (-1, 3), order='F').transpose()
    # np.vstack(...) 将三维点集threedee转换到齐次坐标系下（表现为加了全为1 的第四行, threedee从3×50176变为4×50176）
    # proj_matrix是 3×4 的相机矩阵, 与齐次坐标下的三维点集相乘, 转化为齐次坐标下的二维点集（temp_proj, 3×50176）
    temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
    # numpy.tile() 把数组沿各个方向复制; np.divide() 将每一项做除法, 将齐次坐标转换回笛卡尔坐标（project, 2×50176）
    # project 存储着经人脸对齐后, 224×224的投影视图各个坐标在输入图像中的对应坐标位置
    # project 中存在非正数或超出输入图像宽或高（matlab index）的坐标值时, 表示投影视图的对应索引的点投影在输入图像外
    project = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2,1)))
    ## Getting only the face for debug purpose and the background mask as well
    # 得到背景区域的点的索引集
    bg_mask = np.setdiff1d( np.arange(0, ref_U.shape[0]*ref_U.shape[1]) ,facemask[:,0] )
    # 得到脸部的点的坐标集
    face_proj = project[:, facemask[:,0] ]
    #out_proj = project[:, bg_mask]
    ## Getting points that are outside the image(1×50176)
    bad = badIndex(project, img)
    # nonbadind 存放投影视图所有投影在输入图像内的点的索引序列
    nonbadind = np.nonzero(bad == 0)[0]
    badind = np.nonzero(bad == 1)[0]
    ## Check which points lie outside of the image
    out_proj = project[:, badind]
    out_proj_disp = out_proj
    ind_all = np.arange(0, ref_U.shape[0]*ref_U.shape[1])
    ind_outside = ind_all[badind]
    ############## OUTSIDE ##################################################
    background_img = None
    badface = badIndex(face_proj, img)
    face_in = np.nonzero( badface == 0 )[0]
    face_proj_in = face_proj[:,face_in]
    ## In case we have some points outside, handle the bg
    # 假如有一些点投影到输入图像外, 则处理背景
    out_proj = HandleBackground(out_proj,face_proj_in, img, opts)
    ############## END OUTSIDE ##################################################
    ############## INSIDE ##################################################
    in_proj = project[:, nonbadind]
    # because python arrays are zero indexed
    in_proj -= 1 # matlab indexing
    ind_frontal = ind_all[nonbadind]
    ############## END INSIDE ##################################################

    # 个人添加：
    # displayProj(img, out_proj, in_proj, face_proj_in)

    # To do all at once
    # 合并矩阵
    prj_jnt = np.hstack( (out_proj, in_proj) )
    ind_jnt = np.hstack( (ind_outside, ind_frontal) )

    if opts.getboolean('renderer', 'background'):
        frontal_raw = warpImg(img, ref_U.shape[0], ref_U.shape[1], prj_jnt, ind_jnt)
    else:
        frontal_raw = warpImg(img, ref_U.shape[0], ref_U.shape[1], face_proj, facemask[:,0])

    ## Apply soft-sym if needed
    frontal_sym, sym_weight = mysoftSymmetry(img, frontal_raw, ref_U, in_proj, ind_frontal, bg_mask, facemask[:,0], eyemask, opts)

    return frontal_raw, frontal_sym, face_proj_in, out_proj_disp, out_proj, sym_weight

""" @brief 使用软件形式的边缘融合技术处理输出视图（只处理正脸输出图像, 具有 eyemask）
    :param img 输入图像
    :param frontal_raw 已绘制的输出视图
    :param ref_U 三维头部模型的坐标集（包含脸部和背景区域等所有点的坐标）
    :param in_proj 成像平面内的点集
    :param ind_frontal 投影到输入图像内的点的索引集
    :param bg_mask 背景区域的点的索引集
    :param eyemask 眼睛区域的点的索引集
    :param opts 配置文件对象
    :return frontal_sym 经过边缘融合技术处理后的输出视图
    :return weights 软混合权重
"""
def mysoftSymmetry(img, frontal_raw, ref_U, in_proj, \
                 ind_frontal, bg_mask,facemask, eyemask, opts):
    weights = None
    ## Eyemask is activate only for frontal so we do soft-sym only on frontal thus when we have eyemask
    if eyemask is not None and opts.getboolean('renderer', 'symmetry'): # one side is ocluded
        ## Soft Symmetry param
        ksize_acc = Params.ksize_acc
        ################
        ## SOFT SYMMETRY 
        ind = np.ravel_multi_index((np.asarray(in_proj[1, :].round(), dtype='int64'), np.asarray(in_proj[0, :].round(),
                                    dtype='int64')), dims=img.shape[:-1], order='F')
        synth_frontal_acc = np.zeros(ref_U.shape[:-1])
        c, ic = np.unique(ind, return_inverse=True)
        bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
        count, bin_edges = np.histogram(ind, bin_edges)
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
        synth_frontal_acc[ind_frontal] = count[ic]
        synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
        synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (ksize_acc, ksize_acc), 30., borderType=cv2.BORDER_REPLICATE)
        ## Checking which side has more occlusions?
        midcolumn = np.round(ref_U.shape[1]/2)
        # apply soft symmetry to use whatever parts are visible in ocluded side
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
        minacc=synth_frontal_acc[facemask].min()
        maxacc=synth_frontal_acc[facemask].max()
        ## we may need to do something more smooth like in previous softSym()
        synth_frontal_acc[facemask] = (synth_frontal_acc[facemask] - minacc)/(maxacc-minacc)

        if opts.getboolean('symmetry', 'flipBackground'):
            synth_frontal_acc[bg_mask] = 1. #this control sym on/off on background
        else:
            synth_frontal_acc[bg_mask] = 0. #this control sym on/off on background

        synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')

        synth_frontal_acc = np.tile(synth_frontal_acc.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
        ## Flipping
        frontal_flip = frontal_raw.copy()
        frontal_flip[:,0:midcolumn,:] = np.fliplr(frontal_flip)[:,0:midcolumn,:]
        frontal_sym = np.multiply(frontal_raw, 1.-synth_frontal_acc) + np.multiply(frontal_flip, synth_frontal_acc)


        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype('uint8')
        weights = synth_frontal_acc[:,:,0]
    else: # both sides are occluded pretty much to the same extent -- do not use symmetry
        print '> skipping sym'
        frontal_sym = frontal_raw
    return frontal_sym, weights

#############################################
##### Old Symmetry code ###################################
##################################################
# def softSymmetry(img, frontal_raw, ref_U, in_proj, \
#                  ind_frontal, bg_mask, eyemask, opts):
#     ## Eyemask is activate only for frontal so we do soft-sym only on frontal thus when we have eyemask
#     if eyemask is not None and opts.getboolean('renderer', 'symmetry'): # one side is ocluded
#         ## Soft Symmetry param
#         ACC_CONST = Params.ACC_CONST
#         ksize_acc = Params.ksize_acc
#         ksize_weight = Params.ksize_weight
#         ################
#         ## SOFT SYMMETRY 
#         ind = np.ravel_multi_index((np.asarray(in_proj[1, :].round(), dtype='int64'), np.asarray(in_proj[0, :].round(),
#                                     dtype='int64')), dims=img.shape[:-1], order='F')
#         synth_frontal_acc = np.zeros(ref_U.shape[:-1])
#         c, ic = np.unique(ind, return_inverse=True)
#         bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
#         count, bin_edges = np.histogram(ind, bin_edges)
#         synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
#         synth_frontal_acc[ind_frontal] = count[ic]
#         synth_frontal_acc = synth_frontal_acc.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
#         synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (ksize_acc, ksize_acc), 30., borderType=cv2.BORDER_REPLICATE)
#         ## Checking which side has more occlusions?
#         midcolumn = np.round(ref_U.shape[1]/2)
#         sumaccs = synth_frontal_acc.sum(axis=0)
#         sum_left = sumaccs[0:midcolumn].sum()
#         sum_right = sumaccs[midcolumn+1:].sum()
#         sum_diff = sum_left - sum_right
#         #print '----------------->np.abs(sum_diff), ',  np.abs(sum_diff)
#         if np.abs(sum_diff) > ACC_CONST:
#             print '> Using Face symmetry'
#             ones = np.ones((ref_U.shape[0], midcolumn))
#             zeros = np.zeros((ref_U.shape[0], ref_U.shape[1]-midcolumn))
#             if sum_diff > ACC_CONST: # left side of face has more occlusions
#                 weights = np.hstack((zeros, ones))
#             else: # right side of face has more occlusions
#                 weights = np.hstack((ones, zeros))
#             weights = cv2.GaussianBlur(weights, (ksize_weight, ksize_weight), 60.5, borderType=cv2.BORDER_REPLICATE)

#             # apply soft symmetry to use whatever parts are visible in ocluded side
#             synth_frontal_acc /= synth_frontal_acc.max()
#             weight_take_from_org = 1. / np.exp( synth_frontal_acc )

#             ### This to avoid symmetry in the background
#             #Symmetry only on the face, on the background we simply copy pase the other part
#             weight_take_from_org = weight_take_from_org.reshape(-1, order='F')
#             if opts.getboolean('symmetry', 'flipBackground'):
#                 weight_take_from_org[bg_mask] = 0. #this control sym on/off on background
#             else:
#                 weight_take_from_org[bg_mask] = 1. #this control sym on/off on background
#             weight_take_from_org = weight_take_from_org.reshape((ref_U.shape[0], ref_U.shape[1]), order='F')
#             ###############
#             weight_take_from_sym = 1 - weight_take_from_org
#             #print 'weight_take_from_org.shape,',  weight_take_from_org.shape
#             #print 'weights,',  np.fliplr(weights).shape
#             weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
#             weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

#             weight_take_from_org = np.tile(weight_take_from_org.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
#             weight_take_from_sym = np.tile(weight_take_from_sym.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))
#             weights = np.tile(weights.reshape(ref_U.shape[0], ref_U.shape[1], 1), (1, 1, 3))

#             denominator = weights + weight_take_from_org + weight_take_from_sym
#             frontal_sym =    np.multiply(frontal_raw, weights) +\
#                              np.multiply(frontal_raw, weight_take_from_org) +\
#                              np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
#             frontal_sym = np.divide(frontal_sym, denominator)
#             ## Eye-Mask
#             #frontal_sym = np.multiply(frontal_sym, 1-eyemask) + np.multiply(frontal_raw, eyemask)
#             #########################################
#             frontal_sym[frontal_sym > 255] = 255
#             frontal_sym[frontal_sym < 0] = 0
#             frontal_sym = frontal_sym.astype('uint8')
#         else:
#             print '> not occluded, not doing sym'
#             frontal_sym = frontal_raw
#     else: # both sides are occluded pretty much to the same extent -- do not use symmetry
#         print '> skipping sym'
#         frontal_sym = frontal_raw
#     return frontal_sym

""" @brief 用颜色显示显示投影（个人添加）
    :param img
    :param out_proj
    :param in_proj
    :param face_proj_in
    :param mapping
    :param read
"""
def displayProj(img, out_proj, in_proj, face_proj_in, mapping = True, write = False):
    image = np.zeros((230L, 230L, 3L), dtype=np.uint8)

    for i in range(0, out_proj.shape[1]):
        if int(out_proj[0, i]) < 230 and int(out_proj[1, i]) < 230:
            if mapping:
                image[int(out_proj[1, i]), int(out_proj[0, i])] = img[int(out_proj[1, i]), int(out_proj[0, i])]
            else:
                image[int(out_proj[1, i]), int(out_proj[0, i])] = [0, 255, 0]
    for i in range(0, in_proj.shape[1]):
        if mapping:
            image[int(in_proj[1, i]), int(in_proj[0, i])] = img[int(in_proj[1, i]), int(in_proj[0, i])]
        else:
            image[int(in_proj[1, i]), int(in_proj[0, i])] = [255, 0, 0]
    for i in range(0, face_proj_in.shape[1]):
        if mapping:
           image[int(face_proj_in[1, i]), int(face_proj_in[0, i])] = img[int(face_proj_in[1, i]), int(face_proj_in[0, i])]
        else:
           image[int(face_proj_in[1, i]), int(face_proj_in[0, i])] = [0, 0, 255]

    if write:
        if mapping:
            cv2.imwrite('code_references/render/mapping/color/color' + str(random.uniform(0,10000000)) + '.png', image)
        else:
            cv2.imwrite('code_references/render/mapping/BRG/BRG' + str(random.uniform(0,10000000)) + '.png', image)
    else:
        cv2.imshow('img', image)
        cv2.waitKey()