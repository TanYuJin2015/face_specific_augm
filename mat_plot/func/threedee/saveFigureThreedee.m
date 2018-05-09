function saveFigureThreedee(matFilename, ~)  
    path = 'E:\AppServe\www\face_specific_augm-master-py2\code_references\models3D_mat\model3D.threedee\no-deal\';

    threedee = loadThreedee(matFilename);
    
    [threedee_height, ~] = size(threedee);
    reshape_width = 1;
    reshape_height = threedee_height;
    
    for divide = 3: threedee_height /2 -1
        if mod(threedee, divide) == 0
            reshape_width = divide;
            reshape_height = threedee_height / divide;
        end
    end
    
    x = reshape(threedee(:, 1), [reshape_height, reshape_width]);
    y = reshape(threedee(:, 2), [reshape_height, reshape_width]);
    z = reshape(threedee(:, 3), [reshape_height, reshape_width]);

    fig = figure('Visible', 'off');
    plot3(x, y, z, 'o');
    grid on;
    xlabel('X轴');
    ylabel('Y轴');
    zlabel('Z轴');
    figureTitle = [matFilename(20) , matFilename(21) ,'号头部模型(threedee), ', ...
                    matFilename(13), matFilename(14), matFilename(15), '°'];
    title(figureTitle);
    
    saveas(fig, [path, matFilename, '.png']);
    