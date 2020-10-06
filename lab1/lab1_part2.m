


%% 2.6 Undoing Perspective Distortion of Planar Surface

%% (a)
P = imread('images/book.jpg');
who P
figure, imshow(P);

%% b
% [X, Y] = ginput(4);
x1 = 143; y1 = 28;
x2 = 309; y2 = 48;
x3 = 7; y3 = 160;
x4 = 255; y4 = 213;


X = [143; 309; 7; 255] % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Y = [28; 48; 160; 213]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

Xim = [0; 210; 210; 0]; % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Yim = [0; 0; 297; 297]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

%% c
A = [
    [x1, y1, 1, 0, 0, 0, -Xim(1)*x1, -Xim(1)*y1];
    [0, 0, 0, x1, y1, 1, -Yim(1)*x1, -Yim(1)*y1];
    [x2, y2, 1, 0, 0, 0, -Xim(2)*x2, -Xim(2)*y2];
    [0, 0, 0, x2, y2, 1, -Yim(2)*x2, -Yim(2)*y2];
    [x3, y3, 1, 0, 0, 0, -Xim(3)*x3, -Xim(3)*y3];
    [0, 0, 0, x3, y3, 1, -Yim(3)*x3, -Yim(3)*y3];
    [x4, y4, 1, 0, 0, 0, -Xim(4)*x4, -Xim(4)*y4];
    [0, 0, 0, x4, y4, 1, -Yim(4)*x4, -Yim(4)*y4];
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