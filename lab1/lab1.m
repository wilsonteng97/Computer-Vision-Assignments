%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.1 Contrast Stretching

%% (a) Load image and transform to grayscale
Pc = imread('images/mrt-train.jpg');
whos Pc;
P = rgb2gray(Pc);

%% (b) Display grayscale image
whos P;
figure('Name', 'Grayscale', 'Color', '#D3D3D3'), imshow(P);

%% (c) Find Min & Max intensities of grayscale image
min_intensity = min(P(:)) % Used as offset
max_intensity = max(P(:))

%% d

% norm_factor = 255 / (double(max_intensity) - double(min_intensity));
% normalised_img = (double(P) - double(min_intensity)) * norm_factor;
% P2 = uint8(normalised_img);

% Using Image Processing Toolbox Library
P2(:,:) = imsubtract(P(:,:), double(min_intensity));
P2(:,:) = immultiply(P2(:,:), 255 / (double(max_intensity) - double(min_intensity)));

assert(min(P2(:)) == 0 && max(P2(:)) == 255) % Check if P2 has gone through contrast stretching
disp("Assertion passed : P2 max and min values are 0 and 255 respectively.");
%% e
figure('Name', 'Normalised', 'Color', '#D3D3D3'), imshow(P2, []);
figure('Name', 'Original', 'Color', '#D3D3D3'), imshow(P, []);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.2 Histogram Equalization

%% a
Pc = imread('images/mrt-train.jpg');
P = rgb2gray(Pc);
figure('Name', 'Histogram (10 bins)', 'Color', '#D3D3D3'), imhist(P2, 10);
figure('Name', 'Histogram (256 bins)', 'Color', '#D3D3D3'), imhist(P2, 256);

%% b
P3 = histeq(P, 255);
imhist(P3, 10);
imhist(P3, 256);

%% c
P3 = histeq(P3, 255);
imhist(P3, 10);
imhist(P3, 256);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.3 Linear Spatial Filtering

%% a i)
x = -2:2;
y = -2:2;
sigma_1 = 1.0;

[X,Y] = meshgrid(x,y);

filter_1 = exp(-((X).^2 + (Y).^2) / (2 * sigma_1^2));
filter_1 = filter_1 ./ (2 * pi * sigma_1^2);
filter_1 = filter_1 ./ sum(filter_1(:));
mesh(filter_1);

%% a ii)
x = -2:2;
y = -2:2;
sigma_2 = 2.0;

[X,Y] = meshgrid(x,y);

filter_2 = exp(-((X).^2 + (Y).^2) / (2 * sigma_2^2));
filter_2 = filter_2 ./ (2 * pi * sigma_2^2);
filter_2 = filter_2 ./ sum(filter_2(:));
mesh(filter_2);

%% b
P = imread('image/lib-gn.jpg');
imshow(P);

%% c
P1 = uint8(conv2(P,filter_1));
imshow(P1);
P2 = uint8(conv2(P,filter_2));
imshow(P2);

%% d
P = imread('image/lib-sp.jpg');
imshow(P);

%% e
P1 = uint8(conv2(P,filter_1));
imshow(P1);
P2 = uint8(conv2(P,filter_2));
imshow(P2);

%% Conclusion
% Gaussian filter is more effective in removing gaussian noise than speckle noise.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.4 Median Filtering

%% b
P = imread('image/lib-gn.jpg');
imshow(P);

%% c
P1 = medfilt2(P,[3,3]);
imshow(P1);
P2 = medfilt2(P,[5,5]);
imshow(P2);

%% d 
P = imread('image/lib-sp.jpg');
imshow(P);

%% e
P1 = medfilt2(P,[3,3]);
imshow(P1);
P2 = medfilt2(P,[5,5]);
imshow(P2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.5 Suppressing Noise Interference Patterns

%% a
P = imread('image/pck-int.jpg');
imshow(P);

%% b
F = fft2(P);
S = abs(F);
imagesc(fftshift(S.^0.1));
colormap('default');

%% c
imagesc(S.^0.1);
colormap('default');

%% d
x1 = 241; y1 = 9;
x2 = 17; y2 = 249;
F(x1-2:x1+2, y1-2:y1+2) = 0;
F(x2-2:x2+2, y2-2:y2+2) = 0;
S = abs(F);
imagesc(fftshift(S.^0.1));
colormap('default');

%% e
result = uint8(ifft2(F));
imshow(result);

%% f
P = imread('image/primate-caged.jpg');
P = rgb2gray(P);
imshow(P);

F = fft2(P);    
S = abs(F);
imagesc(S.^0.0001);
colormap('default');

x1 = 252; y1 = 11; F(x1-2:x1+2, y1-2:y1+2) = 0;
x2 = 248; y2 = 22; F(x2-2:x2+2, y2-2:y2+2) = 0;
x3 = 5; y3 = 247; F(x3-2:x3+2, y3-2:y3+2) = 0;
x4 = 10; y4 = 236; F(x4-2:x4+2, y4-2:y4+2) = 0;
S = abs(F);
imagesc(S.^0.1);
colormap('default');

result = uint8(ifft2(F));
imshow(result);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.6 Undoing Perspective Distortion of Planar Surface

%% a
P = imread('image/book.jpg');
imshow(P);

%% b
[X, Y] = ginput(4);
Xim = [0; 210; 210; 0];
Yim = [0; 0; 297; 297];

%% c
A = [
    [X(1), Y(1), 1, 0, 0, 0, -Xim(1)*X(1), -Xim(1)*Y(1)];
    [0, 0, 0, X(1), Y(1), 1, -Yim(1)*X(1), -Yim(1)*Y(1)];
    [X(2), Y(2), 1, 0, 0, 0, -Xim(2)*X(2), -Xim(2)*Y(2)];
    [0, 0, 0, X(2), Y(2), 1, -Yim(2)*X(2), -Yim(2)*Y(2)];
    [X(3), Y(3), 1, 0, 0, 0, -Xim(3)*X(3), -Xim(3)*Y(3)];
    [0, 0, 0, X(3), Y(3), 1, -Yim(3)*X(3), -Yim(3)*Y(3)];
    [X(4), Y(4), 1, 0, 0, 0, -Xim(4)*X(4), -Xim(4)*Y(4)];
    [0, 0, 0, X(4), Y(4), 1, -Yim(4)*X(4), -Yim(4)*Y(4)];
];
v = [Xim(1); Yim(1); Xim(2); Yim(2); Xim(3); Yim(3); Xim(4); Yim(4)];
u = A \ v;
U = reshape([u;1], 3, 3)'; 
w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));

%% d
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

%% e
imshow(P2);