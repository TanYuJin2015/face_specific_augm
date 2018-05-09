function threedee = loadThreedee(matFilename)
    entirePath = ['E:\AppServe\www\face_specific_augm-master-py2\models3d_new\', matFilename];
    
    model3D = load(entirePath);
    
    threedee = model3D.model3D.threedee;