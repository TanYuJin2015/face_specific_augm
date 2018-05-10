function showFigureRefU(matFilename, figureNum, ~)  
    
    refU = loadRefU(matFilename);

    x = reshape(refU(:, :, 1), [224, 224]);
    y = reshape(refU(:, :, 2), [224, 224]);
    z = reshape(refU(:, :, 3), [224, 224]);

    figure(figureNum);
    if nargin == 2
        surfl(x, y, z);  
        xlabel('X÷·');
        ylabel('Y÷·');
        zlabel('Z÷·');
    end
    if nargin == 3
        surfl(z, x, -y);     
        axis square;
        xlabel('Z÷·');
        ylabel('X÷·');
        zlabel('Y÷·');
    end
    figureTitle = [matFilename(20) , matFilename(21) ,'∫≈Õ∑≤øƒ£–Õ(refU), ', ...
                    matFilename(13), matFilename(14), matFilename(15), '°„'];
    title(figureTitle);