%% 2.5 Suppressing Noise Interference Patterns

%% a
P = imread('images/pck-int.jpg');
figure, imshow(P);

%% b
F = fft2(P);
S = abs(F);
imagesc(fftshift(S.^0.1));
figure, colormap('default');

%% c
imagesc(S.^0.1);
figure, colormap('default');

%% d
x1 = 241; y1 = 9;
x2 = 17; y2 = 249;
F(x1-2:x1+2, y1-2:y1+2) = 0;
F(x2-2:x2+2, y2-2:y2+2) = 0;
S = abs(F);
imagesc(fftshift(S.^0.1));
figure, colormap('default');

%% e
result = uint8(ifft2(F));
figure, imshow(result);

%% f
P = imread('images/primate-caged.jpg');
P = rgb2gray(P);
figure, imshow(P);

F = fft2(P);    
S = abs(F);
imagesc(S.^0.0001);
figure, colormap('default');

x1 = 252; y1 = 11; F(x1-2:x1+2, y1-2:y1+2) = 0;
x2 = 248; y2 = 22; F(x2-2:x2+2, y2-2:y2+2) = 0;
x3 = 5; y3 = 247; F(x3-2:x3+2, y3-2:y3+2) = 0;
x4 = 10; y4 = 236; F(x4-2:x4+2, y4-2:y4+2) = 0;
S = abs(F);
imagesc(S.^0.1);
figure, colormap('default');

result = uint8(ifft2(F));
figure, imshow(result);


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
P4e2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

%% e
imshow(P4e2);