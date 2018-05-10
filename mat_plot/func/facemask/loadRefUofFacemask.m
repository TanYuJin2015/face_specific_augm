function refU = loadRefU(matFilename)
    entirePath = ['E:\AppServe\www\face_specific_augm-master-py2\models3d_new\', matFilename];
    
    model3D = load(entirePath);
    
    refU = model3D.model3D.refU;