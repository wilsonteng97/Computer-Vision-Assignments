function [ret] = disparityMap(img_left, img_right, temp_x, temp_y)
%DISPARITYMAP Summary of this function goes here
% Description
% (i) Extract a template comprising the 11x11 neighbourhood region 
%     around that pixel.
% (ii) carry out SSD matching in Pr, but only along the same scanline.
% (iii) Input the disparity into the disparity map with the same Pl pixel 
%       coordinates. 
    tic();
    
    img_left = im2double(img_left); img_right = im2double(img_right);
    [height, width] = size(img_left); 
    [img_right_height, img_right_width] = size(img_right);

    % ensure both images have the same dimensions
    assert(width == img_right_width)
    assert(height == img_right_height)
    % ensure template x size == template y size
    assert(temp_x == temp_y)
    
    % calculate half size of template
    offset = floor(temp_x/2);    
    % initialize
    searchRng = 15;
    % initialize ret
    ret = zeros(size(img_left));

    % for each row of pixels
    for (row = 1 : height)
        % set min/max row bounds for the template and blocks.
        min_row = max(1, row - offset);
        max_row = min(height, row + offset);

        % for each column of pixels
        for (col = 1 : width)
            % Set the min/max column bounds for the template.
            min_col = max(1, col - offset);
            max_col = min(width, col + offset);
            % Define search boundary limits from current (row, col)
            % 'mind' & 'maxd' is the the max displacement of pixels 
            % we can search to the left and right repectively.
            left_displacement = max(-searchRng, 1 - min_col); 
            right_displacement = min(searchRng, width - max_col);
            % Select the block from the right image to use as template.
            template = img_right(min_row:max_row, min_col:max_col);

            ssd_min = inf;
            smallestDiffIndex = 0;
            % Calculate the difference for each of the blocks.
            for i = left_displacement : right_displacement
                % Select a block from the left image at displacement 'i'.
                block = img_left(min_row:max_row, ...
                    (min_col + i):(max_col + i));
                % Compute the sum of squared difference (SSD)
                ssd = sum(sum((template - block).^2));
                
                % FFT implementation but too slow
                % block_transposed = rot90(block,2);
                % ssd_1 = ifft2(fft2(block, temp_y, temp_x).*fft2(block_transposed, temp_y, temp_x));
                % ssd_1 = ssd_1(11,11);
                % ssd_2 = ifft2(fft2(template, temp_y, temp_x).*fft2(block_transposed, temp_y, temp_x));
                % ssd_2 = ssd_2(11, 11);
                % ssd = ssd_1 - 2 * ssd_2;
                
                if ssd < ssd_min
                    ssd_min = ssd;
                    smallestDiffIndex = i - left_displacement + 1;
                end
            end
   
            % Disparity value produced by templates matching.
            ret(row, col) = smallestDiffIndex + left_displacement - 1;
        end

        % Update progress every 10th row.
        if (mod(row, 10) == 0)
            fprintf('  Image row %d / %d done. (%.0f%%)\n', ...
                row, height, (row / height) * 100);
        end

    end
    % Display computation time.
    elapsed = toc();
    fprintf('Calculation %.2f seconds.\n', elapsed);
end





% function [ret] = disparityMap(img_left, img_right, temp_x, temp_y)
% %DISPARITYMAP Summary of this function goes here
% % Description
% % (i) Extract a template comprising the 11x11 neighbourhood region around that pixel.
% % (ii) carry out SSD matching in Pr, but only along the same scanline.
% % (iii) Input the disparity into the disparity map with the same Pl pixel coordinates. 
% 
%     img_left = im2double(img_left);
%     img_right = im2double(img_right);
%     
%     [height, width] = size(img_left);
%     [img_right_height, img_right_width] = size(img_right);
% 
%     % ensure both images have the same dimensions
%     assert(width == img_right_width)
%     assert(height == img_right_height)
% 
%     % calculate half size of template
%     offset_x = floor(temp_x/2);
%     offset_y = floor(temp_y/2);
% 
%     % initialize 
%     ret = ones(height - temp_y + 1, width - temp_x + 1);
% 
%     for row = 1 + offset_y : height - offset_y
%         for col = 1 + offset_x : width - offset_x
%             cur_r = img_left(row - offset_y : row + offset_y, col - offset_x : col + offset_x);
%             cur_l = rot90(cur_r, 2);
%             min_coor = col; 
%             min_diff = inf;
% 
%             % search for simmilar pattern in right image
%             % limit search to 30 pixel to the left
%             for k = max(1 + offset_x , col - 30) : col
%                 T = img_right(row-offset_y: row+offset_y, k-offset_x: k+offset_x);
%                 cur_r = rot90(T, 2);
% 
%                 % Calculate ssd and update minimum diff
%                 conv_1 = conv2(T, cur_r);
%                 conv_2 = conv2(T, cur_l);
%                 ssd = conv_1(temp_x, temp_y) - 2 * conv_2(temp_x, temp_y);
%                 if ssd < min_diff
%                     min_diff = ssd;
%                     min_coor = k;
%                 end
%             end
%             ret(row - offset_x, col - offset_y) = col - min_coor;
%         end
%     end
% end

