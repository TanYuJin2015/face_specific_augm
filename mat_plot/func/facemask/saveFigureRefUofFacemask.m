function saveFigureRefU(matFilename, ~)  
    if nargin == 1
        path = 'E:\AppServe\www\face_specific_augm-master-py2\code_references\models3D_mat\model3D.refU\no-deal\';
    end
    if nargin == 2
        path = 'E:\AppServe\www\face_specific_augm-master-py2\code_references\models3D_mat\model3D.refU\deal\';
    end

    refU = loadRefU(matFilename);
    
    x = reshape(refU(:, :, 1), [224, 224]);
    y = reshape(refU(:, :, 2), [224, 224]);
    z = reshape(refU(:, :, 3), [224, 224]);

    fig = figure('Visible', 'off');
    if nargin == 1
        surfl(x, y, z);  
        xlabel('X÷·');
        ylabel('Y÷·');
        zlabel('Z÷·');
    end
    if nargin == 2
        surfl(z, x, -y);     
        axis square;
        xlabel('Z÷·');
        ylabel('X÷·');
        zlabel('Y÷·');
    end
    figureTitle = [matFilename(20) , matFilename(21) ,'∫≈Õ∑≤øƒ£–Õ(refU), ', ...
                    matFilename(13), matFilename(14), matFilename(15), '°„'];
    title(figureTitle);
    
    saveas(fig, [path, matFilename, '.png']);
    