# -*-coding:utf-8-*
__author__ = 'Douglas'

import urllib2, os, bz2
dlib_facial_landmark_model_url = "http://ufpr.dl.sourceforge.net/project/dclib/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2"

""" @brief 从网络下载 shape_predictor_68_face_landmarks.dat.bz2
    :param url 下载链接
    :param dest dlib模块文件的目录名（"dlib_models"）
"""
def download_file(url, dest):
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(dest+"/"+file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Size: %s (~%4.2fMB)" % (file_name, file_size, (file_size/1024./1024.))

    file_size_dl = 0
    # 8 KB
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        if((file_size_dl*100./file_size) % 5 <= 0.01):
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status
    f.close()
    print "Download complete!"

""" @brief 解压 shape_predictor_68_face_landmarks.dat.bz2 文件 
    :param fpath 要解压的.bz2文件的所在目录
"""
def extract_bz2(fpath):
    print "Extracting..."
    new_file = open(fpath[:-4], "wb")
    file = bz2.BZ2File(fpath, 'rb')
    data = file.read()
    new_file.write(data)
    new_file.close()
    print "Done!"

""" @brief 检查是否已有dlib_models目录和dlib的68点特征点提取的数据文件。
           1. 若没有dlib_models目录，则创建该目录；
           2. 若没有dlib的68点特征点提取的数据文件，则从网络下载对应的.bz2包并解压；若已有则直接本地文件解压。
"""
def check_dlib_landmark_weights():
    dlib_models_folder = "dlib_models"
    if(not os.path.isdir(dlib_models_folder)):
        os.mkdir(dlib_models_folder)
    if(not os.path.isfile(dlib_models_folder+"/shape_predictor_68_face_landmarks.dat")):
        if(not os.path.isfile(dlib_models_folder+"/shape_predictor_68_face_landmarks.dat.bz2")):
            download_file(dlib_facial_landmark_model_url, dlib_models_folder)
        extract_bz2(dlib_models_folder+"/shape_predictor_68_face_landmarks.dat.bz2")