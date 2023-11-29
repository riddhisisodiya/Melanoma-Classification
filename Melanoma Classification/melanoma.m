%% Reading In Relevant Folders
% create image datastore
imds = imageDatastore('lesionimages/');
imgs = readall(imds);   

% create masks datastore
imds_masks = imageDatastore('masks/');
imgs_masks = readall(imds_masks); 

%% Segmentation Using Provided Masks
%{
for i=1:length(imgs_masks)
    binary_mask = imbinarize(imgs_masks{i});
end

for i=1:length(imgs)
    masked_img = imgs{i};
    masked_img(~binary_mask) = 0;
    %histogram_equalisation(imgs{i});
    %change_gamma(imgs{i}, 0.9);
end
%}
%% Call Image Preprocessing Functions
for i=1:length(imgs)
    % LBP
    features = extractLBPFeatures(rgb2gray(imgs{i}));
    q(i, :) = features(:);
    w1 = double(q);

    % asymmetry
    %using mean squared error
    asymmetry = immse(imgs{i}, fliplr(imgs{i}));
    b(i, :) = asymmetry(:);
    x1 = double(b);
    
    % colour
    red = imgs{i}(:, :, 1);
    green = imgs{i}(:, :, 2);
    blue = imgs{i}(:, :, 3);

    white_colour = red == 255 & green == 255 & blue == 255;
    red_w = red(~white_colour);
    green_w = green(~white_colour);
    blue_w = blue(~white_colour);
    colour_summed = horzcat(red_w, green_w, blue_w);

    red_mean = mean(red);
    green_mean = mean(green);
    blue_mean = mean(blue);

    % hog
    hog_features = extractHOGFeatures(imgs{i});

    % glcm
    % texture
    glcms = graycomatrix(rgb2gray(imgs{i}));
    contrast_stats = graycoprops(glcms,{'contrast'});
    homogeneity_stats = graycoprops(glcms,{'homogeneity'});
    correlation_stats = graycoprops(glcms,{'correlation'});
    energy_stats = graycoprops(glcms,{'energy'});

    vect_1 = contrast_stats.Contrast;
    vect_2 = homogeneity_stats.Homogeneity;
    vect_3 = correlation_stats.Correlation;
    vect_4 = energy_stats.Energy;

    glcms_summed = horzcat(vect_1, vect_2, vect_3, vect_4);
end

b4(i, :) = glcms_summed(:);
x4 = double(b4);

b6(i, :) = hog_features(:);
x6 = double(b6);
x66 = squeeze(x6(:, 1));
%% PCA & Colour Histograms
% create colour histograms
for i=1:length(imgs)
 ch = colourhist(imgs{i});
 allhists(i,:) = ch(:); 
end

k = 20;
[one two three] = pca_func(allhists);
four = three(:,1:k);

%% Classification
% concatenating extracted features
p2 = horzcat(four, w1, x1, x4);

imfeatures = [];
imfeatures = [imfeatures; p2];

groundtruth = readcell('groundtruth.txt');
groundtruth(:,1) = [];

% perform classification using 10CV
rng(1); 
svm = fitcsvm(imfeatures, groundtruth);
cvsvm = crossval(svm);
pred = kfoldPredict(cvsvm);
[cm, order] = confusionmat(groundtruth, pred);

%compares true values to predicted
cc = confusionchart(groundtruth, pred);

%% PCA
function [one two three] = pca_func(img6)
% this function performs PCA
    img6 = img6 - mean(img6);         
    C = (img6' * img6)./(size(img6,1)-1);                  
    [one three] = eig(C);
    [three order] = sort(diag(three), 'descend');       
    one = one(:,order);
    two = three(order);
    three = img6 * one;
end

%% Colour Histograms
function H = colourhist(image)
    % function that generates 8x8x8 RGB colour histogram from image
     noBins = 8;  
     binWidth = 256 / noBins; 
     H = zeros(noBins, noBins, noBins); 
    
     [n m d] = size(image);
     data = reshape(image, n*m, d); 
    
     ind = floor(double(data) / binWidth) + 1; 
    
     for i=1:length(ind)
     H(ind(i,1), ind(i,2), ind(i,3)) = H(ind(i,1), ind(i,2), ind(i,3)) + 1; 
     end
     H = H / sum(sum(sum(H))); 
end
%% Image Gamma
function [gam] = change_gamma(img1, val)
% this function changes image gamma
    gam = imadjust(img1, [], [], val);
end

%% Histogram Equalisation
function [hist_eq] = histogram_equalisation(img2)
% this function performs histogram equalisation
    %histogram of image
    h_img2 = imhist(img2);
    
    %Divide by number of pixels
    h_img2_px = h_img2/numel(img2);
    
    %cumsum ew
    cs = cumsum(h_img2_px);
    
    %transformation function
    hist_eq = im2uint8(cs(img2 + 1));
end
 
