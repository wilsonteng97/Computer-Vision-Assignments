%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CZ4003 Computer Vision | Lab Report 1
% Wilson Thurman Teng | U1820540H

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.1 Contrast Stretching

%% (a) Load image and transform to grayscale
Pc = imread('images/mrt-train.jpg');
whos Pc;
P1a = rgb2gray(Pc);
assert(max(max(Pc(:,:,1) - P1a(:,:))) == 0);
assert(max(max(Pc(:,:,2) - P1a(:,:))) == 0);
assert(max(max(Pc(:,:,3) - P1a(:,:))) == 0);

%% (b) Display grayscale image
whos P1a;
figure('Name', 'Grayscale', 'Color', '#D3D3D3');
subplot(1,2,1), imshow(Pc), title('RGB'); % Every pixel has the same value for all 3 channels
subplot(1,2,2), imshow(P1a), title('Grayscale');

%% (c) Find Min & Max intensities of grayscale image
min_intensity = min(P1a(:)) % Used as offset
max_intensity = max(P1a(:))

%% (d) Contrast Stretching

% Native implementation
norm_factor = 255 / (double(max_intensity) - double(min_intensity));
normalised_img = (double(P1a) - double(min_intensity)) * norm_factor;
P1d_ = uint8(normalised_img);

% Using Image Processing Toolbox Library
P1d(:,:) = imsubtract(P1a(:,:), double(min_intensity));
P1d(:,:) = immultiply(P1d(:,:), ...
    255 / (double(max_intensity) - double(min_intensity)));

% Check if P1d and P1d_ has gone through contrast stretching
assert(min(P1d(:)) == 0 && max(P1d(:)) == 255) 
assert(min(P1d_(:)) == 0 && max(P1d_(:)) == 255)
% Check if first principles implmentation is correct
assert(max(max(P1d(:,:) - P1d_(:,:))) == 0); 
disp("Assertion passed : P1d max and min values are 0 and 255 respectively.");

%% (e) Automatic contrast stretching for display

figure('Name', 'Comparison between original and contrast stretched image', 'Color', '#D3D3D3');
subplot(1,2,1), imshow(P1a), title('Original');
subplot(1,2,2), imshow(P1d), title('Contrast Stretched (Normalised)');

% imshow(img, []) displays a contrast stretched image without changing the input matrix. 
figure('Name', 'Comparison between imshow(img) and imshow(img, [])', 'Color', '#D3D3D3');
subplot(1,2,1), imshow(P1a, []), title('Image w/o normalization -> imshow(P1a, [])');
subplot(1,2,2), imshow(P1d), title('Image with normalization -> imshow(P1d)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.2 Histogram Equalization

%% (a) Show histogram of P
Pc = imread('images/mrt-train.jpg');
P2 = rgb2gray(Pc);
figure('Name', 'Histogram Equalization of P2', 'Color', '#D3D3D3');

subplot(3,3,1), imhist(P2, 10), title('Before (10 bins)');
subplot(3,3,2), imhist(P2, 256), title('Before (256 bins)');
subplot(3,3,3), imshow(P2), title('Before');

%% (b) Histogram Equalization on P
P2b = histeq(P2, 256);
subplot(3,3,4), imhist(P2b, 10), title('1st Hist. Equalization (10 bins)');
subplot(3,3,5), imhist(P2b, 256), title('1st Hist. Equalization (256 bins)');
subplot(3,3,6), imshow(P2b), title('After 1st Hist. Equalization');
% 

%% (c) Repeat Histogram Equalization on P
P2c = histeq(P2b, 256);
subplot(3,3,7), imhist(P2c, 10), title('2nd Hist. Equalization (10 bins)');
subplot(3,3,8), imhist(P2c, 256), title('2nd Hist. Equalization (256 bins)');
subplot(3,3,9), imshow(P2c), title('After 2nd Hist. Equalization');

diff = imsubtract(P2b(:,:), P2c(:,:));
figure('Name', 'P2b subtract P2c', 'Color', '#D3D3D3'), imshow(diff), title('Pixels are all 0, indicating P2b == P2c');
% figure('Name', 'Ignore', 'Color', '#D3D3D3'), imshow(diff), title('Ignore');

assert(max(diff(:)) == 0) % Check if P2 has gone through contrast stretching
disp("Assertion passed : P2b == P2c");

figure, imhist(P2, 10), title('Before (10 bins)');
figure, imhist(P2b, 10), title('1st Hist. Equalization (10 bins)');
% Same Histogram even after repeated Histogram Equalization.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.3 Linear Spatial Filtering using a Gaussian PSF (Point Spread Function)

%% (a)(i) 5x5 Gaussian PSF with Sigma = 1.0
% Functions to generate Gaussian PSF
sqrt_dist = @(x,y) ...
    x.^2 + y.^2;
h = @(x, y, sigma) ...
    (1 / (2 * pi * sigma^2)) * (exp(-1*sqrt_dist(x,y) / (2 * sigma^2)));
normalize_h = @(h) ...
    h ./ sum(h(:));

figure('Name', 'Linear Spatial Filtering (Gaussian, 5x5 kernel)', 'Color', '#D3D3D3');

x = -2:2;
y = -2:2;
sigma_a1 = 1.0; % Standard Deviation of distribution.

[X,Y] = meshgrid(x,y);
h1 = h(X, Y, sigma_a1);
h1_norm = normalize_h(h1);
assert(max(max(h1_norm - fspecial('gaussian', 5, 1)))<10^-10)
disp("Assertion passed : h1_norm is correct.");

subplot(1,2,1), mesh(h1_norm), title('Filter h1, sigma=1.0');

%% (a)(ii) 5x5 Gaussian PSF with Sigma = 2.0
x = -2:2;
y = -2:2;
sigma_a2 = 2.0; % Standard Deviation of distribution.

[X,Y] = meshgrid(x,y);
h2 = h(X, Y, sigma_a2);
h2_norm = normalize_h(h2);
assert(max(max(h2_norm - fspecial('gaussian', 5, 2)))<1^-10)
disp("Assertion passed : h2_norm is correct.");

subplot(1,2,2), mesh(h2_norm), title('Filter h2, sigma=2.0');

%% (b) Image with Additive Gaussian Noise
figure('Name', 'lib-gn.jpg - Image with Additive Gaussian Noise', 'Color', '#D3D3D3');
P3b = imread('images/lib-gn.jpg');
subplot(2,2,1), imshow(P3b), title('Original');

%% (c) Effect of Gaussian filter on (b) image
P3c1 = uint8(conv2(P3b, h1, 'same'));
subplot(2,2,3), imshow(P3c1), title('h1 (5x5 Normalized Gaussian, sigma=1.0)');
P3c2 = uint8(conv2(P3b, h2, 'same'));
subplot(2,2,4), imshow(P3c2), title('h2 (5x5 Normalized Gaussian, sigma=2.0)');

%% (d) Image with Additive Speckle Noise
figure('Name', 'lib-sp.jpg - Image with Additive Speckle Noise', 'Color', '#D3D3D3');
P3d = imread('images/lib-sp.jpg');
subplot(2,2,1), imshow(P3d), title('Original');

%% (e) Effect of Gaussian filter on (d) image
P3e1 = uint8(conv2(P3d, h1, 'same'));
subplot(2,2,3), imshow(P3e1), title('h1 (5x5 Normalized Gaussian, sigma=1.0)');
P3e2 = uint8(conv2(P3d, h2, 'same'));
subplot(2,2,4), imshow(P3e2), title('h2 (5x5 Normalized Gaussian, sigma=2.0)');

%% Conclusion
% Gaussian filter is more effective in removing gaussian noise than speckle noise.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.4 Median Filtering

%% (b) Image with Additive Gaussian Noise
figure('Name', 'lib-gn.jpg - Image with Additive Gaussian Noise', 'Color', '#D3D3D3');
P4b = imread('images/lib-gn.jpg');
subplot(2,2,1), imshow(P4b), title('Original');

%% (c) Effect of Median filter on (b) image
% 3x3 median filter kernel
P4c1 = medfilt2(P4b,[3,3]);
subplot(2,2,3), imshow(P4c1), title('Median filter, 3x3');
% 5x5 median filter kernel
P4c2 = medfilt2(P4b,[5,5]);
subplot(2,2,4), imshow(P4c2), title('Median filter, 5x5');

%% (d) Image with Additive Speckle Noise
figure('Name', 'lib-sp.jpg - Image with Additive Speckle Noise', 'Color', '#D3D3D3');
P4d = imread('images/lib-sp.jpg');
subplot(2,2,1), imshow(P4d), title('Original');

%% (e) Effect of Median filter on (d) image
% 3x3 median filter kernel
P4e1 = medfilt2(P4d,[3,3]);
subplot(2,2,3), imshow(P4e1), title('Median filter, 3x3');
% 5x5 median filter kernel
P4e2 = medfilt2(P4d,[5,5]);
subplot(2,2,4), imshow(P4e2), title('Median filter, 5x5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.5 Suppressing Noise Interference Patterns

%% (a) Display pck-int.jpg, an image with dominant diagonal lines
P = imread('images/pck-int.jpg');
figure('Name', '5a'), imshow(P);

%% (b) Obtain Fourier Transform F of image and FFT shift
F = fft2(P); % complex matrix
S = abs(F); % real matrix
figure('Name', '5b'), imagesc(fftshift(log10(S)));
colormap('default');

%% (c) Measure location of peaks on S, w/o FFT shift
figure('Name', '5c'), imagesc(log10(S)); % w/o fftshift
colormap('default');
% Peaks are at (241,9) & (17,249)

%% (d) Remove peaks from F
x1 = 241; 
y1 = 9; 
x2 = 17; 
y2 = 249;

F(x1-2:x1+2, y1-2:y1+2) = 0;
F(x2-2:x2+2, y2-2:y2+2) = 0;
S = abs(F);
figure('Name', '5d'), imagesc(fftshift(log10(S)));
colormap('default');

%% (e)(i) Inverse Fourier Transform on F
result = uint8(ifft2(F));
figure('Name', '5e'), imshow(result);

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
figure('Name', 'Final Result before filter'), imshow(result);

PSF = fspecial('gaussian', 3, 1)
result = uint8(conv2(result, PSF, 'same'));
figure('Name', 'Final Result after filter'), imshow(result, []);

%% (f) "Free" the Primate!
norm = @(array) ...
    array ./ sum(array(:))

P = imread('images/primate-caged.jpg');
P = rgb2gray(P);    
whos P

edges = edge(P, 'Canny' );
selected_edges = bwareaopen(edges,110,8); % Remove objects (8-connected) with less than 110 pixels.
P_copy = P;
P_copy(selected_edges) = 0;
average_pixel_value = sum(P(selected_edges==0)) / (256*256) % Average pixel value of non-edges.

remove_fence = P;
remove_fence(P>=135 & P<=195) = average_pixel_value; % Replace fence with average pixel value of non-edges calculated earlier.
figure('Name', 'Original image VS Fence "removed"'), imshowpair(P, remove_fence, 'montage');

F = fftshift(fft2(P)); % complex matrix
S = abs(F); % real matrix
S_before = S;

norm_S = norm(S);

log10_S = log10(S);
minValue = min(min(log10_S))
maxValue = max(max(log10_S))

amplitudeThreshold = 4.60;
peaks_thres = log10_S > amplitudeThreshold; % Binary image.
% Exclude central peaks. (row 120 to 138)
peaks_thres_central_excluded = peaks_thres;
peaks_thres_central_excluded(:, 122:136) = 0;
figure('Name', 'Peaks from Thresholding'), imshowpair(peaks_thres, peaks_thres_central_excluded, 'montage');

F(peaks_thres_central_excluded) = 0;
S = abs(F);
figure('Name', 'Removed Peaks');
subplot(1,2,1), imagesc(log10(S_before)), title('before');
subplot(1,2,2), imagesc(log10(S)), title('removed peaks');

result = uint8(ifft2(F)); % Inverse fft to obtain image
result(result==0) = average_pixel_value; % Replace dark spots with average pixel value of non-edges calculated earlier.

figure('Name', 'Final result');
subplot(1,2,1), imshow(result, []), title('Before filter');
PSF = fspecial('gaussian', 3, 0.5);
result = uint8(conv2(result, PSF, 'same'));
subplot(1,2,2), imshow(result, []), title('After filter (Final Result)');

figure('Name', 'Final Result'), imshowpair(P, result, 'montage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.6 Undoing Perspective Distortion of Planar Surface

%% (a) Display book.jpg
P = imread('images/book.jpg');
figure, imshow(P), axis on; 
whos P

%% (b)Obtain the corner of the book in the image, as well as the desired 4 corners
% [X, Y] = ginput(4);
x1 = 142; y1 = 28; % Top-left
x2 = 308; y2 = 48; % Top-right
x3 = 3; y3 = 159; % Btm-Left
x4 = 257; y4 = 216; % Btm-right


Xw = [x1; x2; x3; x4]; % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Yw = [y1; y2; y3; y4]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

Xim = [0; 210; 0; 210]; % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Yim = [0; 0; 297; 297]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

%% (c) Setup matrix to estimate the projective transformation 
% Generate matrix A
A = zeros(8,8);
for i = 1:4
    A(i*2-1, :) = [Xw(i); Yw(i); 1; 0; 0; 0; (-1 * Xim(i) * Xw(i)); (-1 *  Xim(i) * Yw(i))];
    A(i*2, :) = [0; 0; 0; Xw(i); Yw(i); 1; (-1 * Yim(i) * Xw(i)); (-1 *  Yim(i) * Yw(i))];
end

v = [Xim(1); Yim(1); Xim(2); Yim(2); Xim(3); Yim(3); Xim(4); Yim(4)];
u = A \ v;
U = reshape([u;1], 3, 3)'; 

% Verify U is correct by transforming into original coordinates.
w = U*[Xw'; Yw'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:))

%% (d) Warp image
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

%% (e) Display image
figure('Name', 'Result');
subplot(1,2,1), imshow(P), title('Original');
subplot(1,2,2), imshow(P2), title('After Warping');