new_model_angle = {'-00', '-22', '-40', '-55', '-70', '-75'};
nSub = 10;

for poseId = 1: 6
    for i = 1: nSub   
        new_model_id = num2str(i, '%02d');
        matFilename = ['model3D_aug_', char(new_model_angle(poseId)), '_00_', new_model_id, '.mat']; 
        
        % 将三维图以.png图像文件形式保存到：
        % E:\AppServe\www\face_specific_augm-master-py2\code_references\models3D_mat\model3D.threedee\no-deal
        saveFigureThreedee(matFilename);
    end
end