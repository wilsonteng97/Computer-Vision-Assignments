


%% 2.6 Undoing Perspective Distortion of Planar Surface

%% (a)
P = imread('images/book.jpg');
%figure, imshow(P), axis on; 
who P

%% b
% [X, Y] = ginput(4);
x1 = 142; y1 = 28; % Top-left
x2 = 308; y2 = 48; % Top-right
x3 = 3; y3 = 159; % Btm-Left
x4 = 257; y4 = 216; % Btm-right


Xw = [x1; x2; x3; x4]; % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Yw = [y1; y2; y3; y4]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

Xim = [0; 210; 0; 210]; % x-coord (Top-left; Top-right; Btm-Left, Btm-right)
Yim = [0; 0; 297; 297]; % y-coord (Top-left; Top-right; Btm-Left, Btm-right)

%% c
A = zeros(8,8);
for i = 1:4
    A(i*2-1, :) = [Xw(i); Yw(i); 1; 0; 0; 0; (-1 * Xim(i) * Xw(i)); (-1 *  Xim(i) * Yw(i))];
    A(i*2, :) = [0; 0; 0; Xw(i); Yw(i); 1; (-1 * Yim(i) * Xw(i)); (-1 *  Yim(i) * Yw(i))];
end

v = [Xim(1); Yim(1); Xim(2); Yim(2); Xim(3); Yim(3); Xim(4); Yim(4)];
u = A \ v;
U = reshape([u;1], 3, 3)'; 
w = U*[Xw'; Yw'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:))

%% d
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

%% e
figure('Name', 'Result');
subplot(1,2,1), imshow(P), title('Original');
subplot(1,2,2), imshow(P2), title('After Warping');