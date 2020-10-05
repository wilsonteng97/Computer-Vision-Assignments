%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.1 Contrast Stretching

%% (a) Load image and transform to grayscale
Pc = imread('images/mrt-train.jpg');
whos Pc;
P1a = rgb2gray(Pc);

%% (b) Display grayscale image
whos P1a;
figure('Name', 'Grayscale', 'Color', '#D3D3D3');
subplot(1,2,1), imshow(Pc), title('RGB'); % Every pixel has the same value for all 3 channels
subplot(1,2,2), imshow(P1a), title('Grayscale');

%% (c) Find Min & Max intensities of grayscale image
min_intensity = min(P1a(:)) % Used as offset
max_intensity = max(P1a(:))

%% d

% norm_factor = 255 / (double(max_intensity) - double(min_intensity));
% normalised_img = (double(P) - double(min_intensity)) * norm_factor;
% P2 = uint8(normalised_img);

% Using Image Processing Toolbox Library
P1d(:,:) = imsubtract(P1a(:,:), double(min_intensity));
P1d(:,:) = immultiply(P1d(:,:), 255 / (double(max_intensity) - double(min_intensity)));

assert(min(P1d(:)) == 0 && max(P1d(:)) == 255) % Check if P2 has gone through contrast stretching
disp("Assertion passed : P1d max and min values are 0 and 255 respectively.");

%% e

figure('Name', 'Comparison between original and contrast stretched image', 'Color', '#D3D3D3');
subplot(1,2,1), imshow(P1a), title('Original');
subplot(1,2,2), imshow(P1d), title('Contrast Stretched (Normalised)');

% imshow(img, []) displays a contrast stretched image without changing the input matrix. 
figure('Name', 'Original (imshow(Pla, []))', 'Color', '#D3D3D3'), imshow(P1a, []), title('With imshow(Pla, [])');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.2 Histogram Equalization

%% (a) Show histogram of P
Pc = imread('images/mrt-train.jpg');
P2 = rgb2gray(Pc);
figure('Name', 'Histogram Equalization of P2', 'Color', '#D3D3D3');

subplot(3,3,1), imhist(P2, 10), title('Before (10 bins)');
subplot(3,3,2), imhist(P2, 256), title('Before (256 bins)');
subplot(3,3,3), imshow(P2), title('Before');

%

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
figure('Name', 'Ignore', 'Color', '#D3D3D3'), imshow(diff), title('Ignore');

assert(max(diff(:)) == 0) % Check if P2 has gone through contrast stretching
disp("Assertion passed : P2b == P2c");

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
assert(max(max(h1_norm - fspecial('gaussian', 5, 1)))<1^-10)
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

