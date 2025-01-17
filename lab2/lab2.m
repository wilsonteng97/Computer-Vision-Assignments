%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CZ4003 Computer Vision | Lab Report 2
% Wilson Thurman Teng | U1820540H

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.1 Edge Detection

%% (a) Load image and transform to grayscale
Pc = imread('images/macritchie.jpg');
whos Pc;
P1a = rgb2gray(Pc);
figure('Name', 'Grayscale img', 'Color', '#D3D3D3'); 
imshow(P1a, []); axis on;
min_intensity = min(P1a(:))
max_intensity = max(P1a(:))

%% (b) Generate Horizontal and vertical edges using Sobel Filter
sobel_h_mask = [
    -1 -2 -1; 
    0 0 0; 
    1 2 1
];
sobel_v_mask = [
    -1 0 1; 
    -2 0 2; 
    -1 0 1
];

% Horizontal edges using Vertical Sobel filter
horizontal = conv2(im2double(P1a), sobel_h_mask, 'same');
horizontal = abs(horizontal);
% Contrast Stretching
min_intensity = min(horizontal(:)) % Used as offset
max_intensity = max(horizontal(:))
horizontal(:,:) = imsubtract(horizontal(:,:), double(min_intensity));
horizontal(:,:) = immultiply(horizontal(:,:), ...
    255 / (double(max_intensity) - double(min_intensity)));

% Vertical edges using Horizontal Sobel filter
vertical = conv2(im2double(P1a), sobel_v_mask, 'same');
vertical = abs(vertical);
% Contrast Stretching
min_intensity = min(vertical(:)) % Used as offset
max_intensity = max(vertical(:))
vertical(:,:) = imsubtract(vertical(:,:), double(min_intensity));
vertical(:,:) = immultiply(vertical(:,:), ...
    255 / (double(max_intensity) - double(min_intensity)));

figure('Name', 'Grayscale', 'Color', '#D3D3D3');
subplot_tight(1,2,1), imshow(uint8(horizontal), []), title('Horizontal edges using Horizontal Sobel mask');
subplot_tight(1,2,2), imshow(uint8(vertical), []), title('Vertical edges using Vertical Sobel mask');

% For edges that are not strictly vertical nor horizontal (i.e. diagonal),
% the edges are fainter than strictly vertical or horizontal edges.

%% (c) Combined Edge Image
combined_edges = sqrt(horizontal.^2 + vertical.^2);
% Contrast Stretching
min_intensity = min(combined_edges(:)) % Used as offset
max_intensity = max(combined_edges(:))
combined_edges(:,:) = imsubtract(combined_edges(:,:), double(min_intensity));
combined_edges(:,:) = immultiply(combined_edges(:,:), ...
    255 / (double(max_intensity) - double(min_intensity)));
figure('Name', 'Combined edges', 'Color', '#D3D3D3');
imshow(uint8(combined_edges), [])

% By squaring the vertical and horizontal gradient approximations,
% summing them up and applying the square root, we obtain the magnitude of
% the gradient of a particular pixel. This therefore allows us to extract
% the edges of the image as pixels that have gradients of large magnitude
% are likly part of an edge in the image. An added benefit is that it
% allows us to extract diagonal edges as both horizontal and vertical
% components are being taken into account now.

%% (d) Thresholding
% Min & Max intensities
combined_edges = uint8(combined_edges);
min_intensity = min(combined_edges(:));
max_intensity = max(combined_edges(:));

% Histogram to determine threshold value
figure('Name', 'Deciding on threshold values', 'Color', '#D3D3D3');
subplot_tight(3,3,1), imshow(combined_edges), title('Original Image');
subplot_tight(3,3,2), imhist(combined_edges, 256), title('Histogram (256 bins)');

% Thresholded images
Et1 = combined_edges > 10;
Et2 = combined_edges > 40;
Et3 = combined_edges > 70;
Et4 = combined_edges > 100;
Et5 = combined_edges > 130;
Et6 = combined_edges > 160;

subplot_tight(3,3,4), imshow(Et1, []), title('Threshold = 10');
subplot_tight(3,3,5), imshow(Et2, []), title('Threshold = 40');
subplot_tight(3,3,6), imshow(Et3, []), title('Threshold = 70');
subplot_tight(3,3,7), imshow(Et4, []), title('Threshold = 100');
subplot_tight(3,3,8), imshow(Et5, []), title('Threshold = 130');
subplot_tight(3,3,9), imshow(Et6, []), title('Threshold = 160');

Et = combined_edges > 50;
figure(), imshow(Et, []), title('Best threshold = 50');

% Low Thresholds :
% + 1) Detailed edge information about the image is retained. This is useful
% if the goal is to detect noise.
% - 1) Low Thresholds are sensitive to noise. This is evident when threshold 
% value is 10 and edges are detected in almost the entire image. 

% High Thresholds :
% + 1) Although still susceptible to noise, we have more accurate edges.
% - 1) If the threshold value is too high, we may not detect any edges at
% all. In the histogram of the macritchie.jpg image, the number of pixels
% with an intensity value of 160 or greater is low. Hence, we were unable
% to see any detected edges with semantic significance when the threshold
% value is set to 160.

%% (e) Canny edge detection algorithm
tl = 0.04; th = 0.1; sigma = 1.0;
cannyedges = edge(P1a, 'canny', [tl th], sigma);
figure('Name', 'Canny edges', 'Color', '#D3D3D3');
sgtitle('Thres = [0.04 0.1], sigma = 1.0');
imshow(cannyedges, [])

%% (e)(i) Canny edge detection algorithm (Varying Sigma)
tl = 0.04; th = 0.35;
cannyedges_sigma1 = edge(P1a, 'canny', [tl th], 1.0);
cannyedges_sigma15 = edge(P1a, 'canny', [tl th], 1.5);
cannyedges_sigma2 = edge(P1a, 'canny', [tl th], 2.0);
cannyedges_sigma25 = edge(P1a, 'canny', [tl th], 2.5);
cannyedges_sigma3 = edge(P1a, 'canny', [tl th], 3.0);
cannyedges_sigma35 = edge(P1a, 'canny', [tl th], 3.5);
cannyedges_sigma4 = edge(P1a, 'canny', [tl th], 4.0);
cannyedges_sigma45 = edge(P1a, 'canny', [tl th], 4.5);
cannyedges_sigma5 = edge(P1a, 'canny', [tl th], 5.0);

figure('Name', 'Canny edges, changing sigma (1 - 3)', 'Color', '#D3D3D3');
sgtitle('Thres = [0.04 0.35], sigma = ?');
subplot_tight(1,4,1), imshow(cannyedges_sigma1, []), title('Sigma = 1.0');
subplot_tight(1,4,2), imshow(cannyedges_sigma15, []), title('Sigma = 1.5');
subplot_tight(1,4,3), imshow(cannyedges_sigma2, []), title('Sigma = 2.0');
subplot_tight(1,4,4), imshow(cannyedges_sigma25, []), title('Sigma = 2.5');

figure('Name', 'Canny edges, changing sigma (3.5 - 5)', 'Color', '#D3D3D3');
subplot_tight(1,5,1), imshow(cannyedges_sigma3, []), title('Sigma = 3.0');
subplot_tight(1,5,2), imshow(cannyedges_sigma35, []), title('Sigma = 3.5');
subplot_tight(1,5,3), imshow(cannyedges_sigma4, []), title('Sigma = 4.0');
subplot_tight(1,5,4), imshow(cannyedges_sigma45, []), title('Sigma = 4.5');
subplot_tight(1,5,5), imshow(cannyedges_sigma5, []), title('Sigma = 5.0');

%% (e)(ii) Canny edge detection algorithm (Varying Threshold Low)
th = 0.35; sigma = 1.0; 
cannyedges_sigma0 = edge(P1a, 'canny', [0.00 th], sigma);
cannyedges_sigma1 = edge(P1a, 'canny', [0.01 th], sigma);
cannyedges_sigma2 = edge(P1a, 'canny', [0.02 th], sigma);
cannyedges_sigma3 = edge(P1a, 'canny', [0.03 th], sigma);
cannyedges_sigma4 = edge(P1a, 'canny', [0.04 th], sigma);

cannyedges_sigma5 = edge(P1a, 'canny', [0.05 th], sigma);
cannyedges_sigma6 = edge(P1a, 'canny', [0.06 th], sigma);
cannyedges_sigma7 = edge(P1a, 'canny', [0.07 th], sigma);
cannyedges_sigma8 = edge(P1a, 'canny', [0.08 th], sigma);
cannyedges_sigma9 = edge(P1a, 'canny', [0.09 th], sigma);

figure('Name', 'Canny edges, changing threshold low (0 - 0.04)', 'Color', '#D3D3D3');
sgtitle('Thres = [? 0.35], sigma = 1.0');
subplot_tight(1,5,1), imshow(cannyedges_sigma0, []), title('Thres Low = 0.00');
subplot_tight(1,5,2), imshow(cannyedges_sigma1, []), title('Thres Low = 0.01');
subplot_tight(1,5,3), imshow(cannyedges_sigma2, []), title('Thres Low = 0.02');
subplot_tight(1,5,4), imshow(cannyedges_sigma3, []), title('Thres Low = 0.03');
subplot_tight(1,5,5), imshow(cannyedges_sigma4, []), title('Thres Low = 0.04');

figure('Name', 'Canny edges, changing threshold low (0.05 - 0.09)', 'Color', '#D3D3D3');
subplot_tight(1,5,1), imshow(cannyedges_sigma5, []), title('Thres Low = 0.05');
subplot_tight(1,5,2), imshow(cannyedges_sigma6, []), title('Thres Low = 0.06');
subplot_tight(1,5,3), imshow(cannyedges_sigma7, []), title('Thres Low = 0.07');
subplot_tight(1,5,4), imshow(cannyedges_sigma8, []), title('Thres Low = 0.08');
subplot_tight(1,5,5), imshow(cannyedges_sigma9, []), title('Thres Low = 0.09');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.2 Line Finding using Hough Transform

%% (a) Reuse canny edges image with tl = 0.04, th = 0.1 and sigma = 1.0
Pc = imread('images/macritchie.jpg');
P2a = rgb2gray(Pc);
whos P2a
% P2a = imread('images/test.png');
tl = 0.04; th = 0.1; sigma = 1.0;
cannyedges = edge(P2a, 'canny', [tl th], sigma);
figure('Name', 'Canny edges', 'Color', '#D3D3D3');
sgtitle('Thres = [0.04 0.1], sigma = 1.0');
imshow(cannyedges, [])

%% (b) Radon Transform (Equivalent to Hough Transform for binary images)
help radon
default_theta = 0:179;
[H, xp] = radon(cannyedges, default_theta); 
H_size = size(H), xp_size = size(xp)
figure('Name', 'Radon Transform on Binary Image', 'Color', '#D3D3D3');
%subplot_tight(1,2,1), 
imagesc(H,'Xdata',default_theta,'Ydata',xp);
title('R_{\theta} (x^{\prime})');
xlabel('\theta (degrees)'); ylabel('x^{\prime} (pixels from center)');
colormap('default'), colorbar


% cannyedges = edge(P2a, 'canny', [tl th], sigma);
% [H2,theta,rho] = hough(cannyedges,'RhoResolution',1,'Theta',-90:0.5:89); 
% size(H2), size(theta), size(rho)
% figure('Name', 'Comparison between Radon & Hough transform', 'Color', '#D3D3D3');
% subplot_tight(1,2,1), imagesc(H,'Xdata',default_theta,'Ydata',xp);
% title('R_{\theta} (x^{\prime})');
% xlabel('\theta (degrees)'); ylabel('x^{\prime} (pixels from center)');
% colormap('default'), colorbar
% subplot_tight(1,2,2), imagesc(H2,'Xdata',theta,'Ydata',rho);
% title('R_{\theta} (\rho)');
% xlabel('\theta'), ylabel('\rho (pixels from coordinate origin 0,0)');
% colormap('default'), colorbar

% 

%% (c) Location of maximum pixel intensity in Hough Image in [theta, radius] form
maxH = max(H(:))
maxH_img = H>=int32(maxH);
figure('Name', 'maximum pixel intensity in Hough Projection', 'Color', '#D3D3D3');
imagesc(maxH_img,'Xdata',default_theta,'Ydata',xp)
title('R_{\theta} (x^{\prime})');
xlabel('\theta (degrees)'); ylabel('x^{\prime} (pixels from center)');
colormap('default'), colorbar

[radiusOfMaxWithoutOffset, thetaOfMaxWithoutOffset] = find(H == maxH) % After accounting for offset, theta:103, radius:-76
radiusOfMax = xp(radiusOfMaxWithoutOffset)
thetaOfMax = thetaOfMaxWithoutOffset - 1

%% (d) Convert the [theta, radius] line representation to the normal line equation Ax + By = C form in image coordinates
% After accounting for offset, thetaOfMax = 103 & radiusOfMax = -76
[img_height, img_width] = size(P1a);
[A, B] = pol2cart(thetaOfMax*pi/180, radiusOfMax);
A, B = -B % B needs to be negated because the y-axis is pointing downwards for image coordinates.

C = A * (A + img_width/2) + B * (B + img_height/2)

%% (e) Find value of y when x = 0 and x = width of image - 1
xl = 0; xr = img_width - 1;
yl = (C - A * xl) / B
yr = (C - A * xr) / B

%% (f) Superimpose estimated line on original image
figure('Name', 'Strongest Edge line', 'Color', '#D3D3D3'); 
subplot_tight(1,2,1), imshow(P1a);
title('Strongest Edge line (theta=103, radius=-76)');
line([xl xr], [yl yr],'Color','red','LineStyle','--')

new_theta_range = 0:0.01:179;
[new_H, xp] = radon(cannyedges, new_theta_range);
[newRadiusOfMaxWithoutOffset, newThetaOfMaxWithoutOffset] = find(new_H == max(new_H(:))) % After accounting for offset, theta:103, radius:-76
newRadiusOfMax = xp(newRadiusOfMaxWithoutOffset)
newThetaOfMax = new_theta_range(newThetaOfMaxWithoutOffset) % new_theta = 102.72
[A, B] = pol2cart(newThetaOfMax*pi/180, newRadiusOfMax);
A, B = -B
C = A * (A + img_width/2) + B * (B + img_height/2)

yl2 = (C - A * xl) / B
yr2 = (C - A * xr) / B
subplot_tight(1,2,2), imshow(P1a);
title('Strongest Edge line (theta=102.72, radius=-76)');
line([xl xr], [yl2 yr2],'Color','red','LineStyle','--')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.3 3D Stereo

% (i) Extract a template comprising the 11x11 neighbourhood region around that pixel.
% (ii) carry out SSD matching in Pr, but only along the same scanline.
% (iii) Input the disparity into the disparity map with the same Pl pixel coordinates. 

%% (a) Disparity map algorithm as a MATLAB function script
% 2 arguments of left and right images
% 2 arguments specifying the template dimensions
% It should return the disparity map.

% function declared in disparityMap.m

%% (b) Download the synthetic stereo pair images of ‘corridorl.jpg’ and ‘corridorr.jpg’, converting both to grayscale.
l = imread('images/corridorl.jpg'); 
l1 = rgb2gray(l);
r = imread('images/corridorr.jpg');
r1 = rgb2gray(r);

% Take red channels from left image
leftRed = l(:,:,1);
% Take the green and blue color channels from the right image.
rightGreenBlue = r(:,:,2:3);
% Combine channels into a single composite image
composite = cat(3, leftRed, rightGreenBlue);

figure('Name', '‘corridorl.jpg’ and ‘corridorr.jpg’', 'Color', '#D3D3D3');
subplot_tight(1,2,1), imshow(l1, []), title('Left Image');
subplot_tight(1,2,2), imshow(r1, []), title('Right Image');

figure('Name', 'corridor composite image', 'Color', '#D3D3D3');
imshow(composite, []), title('Composite image (Corridor)'); axis on;

%% (c) Obtain Disparity Map D
D1 = disparityMap(l1, r1, 11, 11);
res1 = imread('images/corridor_disp.jpg');

figure('Name', '‘Disparity Map of corridor', 'Color', '#D3D3D3');
subplot_tight(1,2,1), imshow(D1,[-15 15]), title('Obtained Disparity Map');
subplot_tight(1,2,2), imshow(res1, []), title('Expected Disparity Map');

%% (d) Rerun algorithm on the real images of ‘triclops-i2l.jpg’ and triclops-i2r.jpg’. 
l = imread('images/triclops-i2l.jpg'); 
l2 = rgb2gray(l);
r = imread('images/triclops-i2r.jpg');
r2 = rgb2gray(r);

D2 = disparityMap(l2, r2, 11, 11);
res2 = imread('images/triclops-id.jpg');

figure('Name', '‘Disparity Map of triclops', 'Color', '#D3D3D3');
subplot_tight(1,2,1), imshow(l2, []), title('Left Image');
subplot_tight(1,2,2), imshow(r2, []), title('Right Image');
figure('Name', '‘Disparity Map of triclops', 'Color', '#D3D3D3');
subplot_tight(1,2,1), imshow(D2, [-15 15]), title('Obtained Disparity Map');
subplot_tight(1,2,2), imshow(res2, []), title('Expected Disparity Map');

% Take red channels from left image
leftRed = l(:,:,1);
% Take the green and blue color channels from the right image.
rightGreenBlue = r(:,:,2:3);
% Combine channels into a single composite image
composite = cat(3, leftRed, rightGreenBlue);
figure('Name', 'triclops composite image', 'Color', '#D3D3D3');
imshow(composite, []), title('Composite image (triclops)'); axis on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3.4 Spatial Pyramid Matching (SPM)

% In Python Notebook

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Utility Function(s)
function vargout=subplot_tight(m, n, p, margins, varargin)
% subplot_tight from Controllable tight subplot library 
% (https://www.mathworks.com/matlabcentral/fileexchange/30884-controllable-tight-subplot)
% A subplot function substitude with margins user tunabble parameter.
%
% Syntax
%  h=subplot_tight(m, n, p);
%  h=subplot_tight(m, n, p, margins);
%  h=subplot_tight(m, n, p, margins, subplotArgs...);
%
% Description
% Our goal is to grant the user the ability to define the margins between neighbouring
%  subplots. Unfotrtunately Matlab subplot function lacks this functionality, and the
%  margins between subplots can reach 40% of figure area, which is pretty lavish. While at
%  the begining the function was implememnted as wrapper function for Matlab function
%  subplot, it was modified due to axes del;etion resulting from what Matlab subplot
%  detected as overlapping. Therefore, the current implmenetation makes no use of Matlab
%  subplot function, using axes instead. This can be problematic, as axis and subplot
%  parameters are quie different. Set isWrapper to "True" to return to wrapper mode, which
%  fully supports subplot format.
%
% Input arguments (defaults exist):
%   margins- two elements vector [vertical,horizontal] defining the margins between
%        neighbouring axes. Default value is 0.04
%
% Output arguments
%   same as subplot- none, or axes handle according to function call.
%
% Issues & Comments
%  - Note that if additional elements are used in order to be passed to subplot, margins
%     parameter must be defined. For default margins value use empty element- [].
%  - 
%
% Example
% close all;
% img=imread('peppers.png');
% figSubplotH=figure('Name', 'subplot');
% figSubplotTightH=figure('Name', 'subplot_tight');
% nElems=17;
% subplotRows=ceil(sqrt(nElems)-1);
% subplotRows=max(1, subplotRows);
% subplotCols=ceil(nElems/subplotRows);
% for iElem=1:nElems
%    figure(figSubplotH);
%    subplot(subplotRows, subplotCols, iElem);
%    imshow(img);
%    figure(figSubplotTightH);
%    subplot_tight(subplotRows, subplotCols, iElem, [0.0001]);
%    imshow(img);
% end
%
% See also
%  - subplot
%
% Revision history
% First version: Nikolay S. 2011-03-29.
% Last update:   Nikolay S. 2012-05-24.
%
% *List of Changes:*
% 2012-05-24
%  Non wrapping mode (based on axes command) added, to deal with an issue of disappearing
%     subplots occuring with massive axes.
% Default params
isWrapper=false;
if (nargin<4) || isempty(margins)
    margins=[0.04,0.04]; % default margins value- 4% of figure
end
if length(margins)==1
    margins(2)=margins;
end
%note n and m are switched as Matlab indexing is column-wise, while subplot indexing is row-wise :(
[subplot_col,subplot_row]=ind2sub([n,m],p);  
height=(1-(m+1)*margins(1))/m; % single subplot height
width=(1-(n+1)*margins(2))/n;  % single subplot width
% note subplot suppors vector p inputs- so a merged subplot of higher dimentions will be created
subplot_cols=1+max(subplot_col)-min(subplot_col); % number of column elements in merged subplot 
subplot_rows=1+max(subplot_row)-min(subplot_row); % number of row elements in merged subplot   
merged_height=subplot_rows*( height+margins(1) )- margins(1);   % merged subplot height
merged_width= subplot_cols*( width +margins(2) )- margins(2);   % merged subplot width
merged_bottom=(m-max(subplot_row))*(height+margins(1)) +margins(1); % merged subplot bottom position
merged_left=min(subplot_col)*(width+margins(2))-width;              % merged subplot left position
pos=[merged_left, merged_bottom, merged_width, merged_height];
if isWrapper
   h=subplot(m, n, p, varargin{:}, 'Units', 'Normalized', 'Position', pos);
else
   h=axes('Position', pos, varargin{:});
end
if nargout==1
   vargout=h;
end
end