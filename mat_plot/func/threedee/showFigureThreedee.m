function showFigureThreedee(matFilename, figureNum)  
    
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

    figure(figureNum);
    plot3(x, y, z, 'o');
    grid on;
    xlabel('X轴');
    ylabel('Y轴');
    zlabel('Z轴');
    figureTitle = [matFilename(20) , matFilename(21) ,'号头部模型(threedee), ', ...
                    matFilename(13), matFilename(14), matFilename(15), '°'];
    title(figureTitle);