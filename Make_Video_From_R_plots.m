
clear all; close all;clc

% vidFile = VideoWriter("Feature_selection_results1.mp4",'MPEG-4')
% vidFile.FrameRate = 1/2; % 1 frame every 2 seconds
% open(vidFile)
% for i = 1:10
%     image= imread(['deviance_run1_fold',num2str(i),'.png']);
%     writeVideo(vidFile,image)
% end
% 


filename = 'Feature_selection_deviance5.gif';
delayTime = 2; % Delay between frames in seconds

for i = 1:10
    [image,map] = imread(['deviance_run5_fold', num2str(i), '.png']);

    
    if i == 1
        imwrite(image, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', delayTime);
    else
        imwrite(image, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
    end
end


filename = 'Feature_selection_hist5.gif';
delayTime = 2; % Delay between frames in seconds

for i = 1:10
    [image,map] = imread(['histogram_run5_fold', num2str(i), '.png']);

    
    if i == 1
        imwrite(image, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', delayTime);
    else
        imwrite(image, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
    end
end
