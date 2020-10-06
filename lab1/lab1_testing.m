%% (e)(ii) Improve upon results from (e)(i)
norm = @(array) ...
    array ./ sum(array(:))

P = imread('images/pck-int.jpg');
whos P
F = fftshift(fft2(P)); % complex matrix
S = abs(F); % real matrix

norm_S = norm(S);

minValue = min(min(S))
maxValue = max(max(S))

amplitudeThreshold = 50000;
peaks = S > amplitudeThreshold; % Binary image.
figure('Name', 'Peaks from Thresholding'), imshow(peaks);
% Exclude the central DC spike. (row 115 to 143)
peaks(118:140, :) = 0;
peaks(:, 125:132) = 0;
figure('Name', 'Exclude central peaks'), imshow(peaks);

F(peaks) = 0;
S = abs(F);
figure('Name', 'Removed Peaks'), imagesc(fftshift(log10(S)));

result = uint8(ifft2(F));
% average_pixels = sum(result(:)) / (256 * 256) * 2;
% result(result==0) = average_pixels;
figure('Name', 'Final Result before filter'), imshow(result);

PSF = fspecial('gaussian', 3, 1)
result = uint8(conv2(result, PSF, 'same'));
figure('Name', 'Final Result after filter'), imshow(result, []);
