%% This script generates the training dataset for the fully convolutional 
% neural network by extracting patches with heatmap labels from the 
% ThuderSTORM simulated data.

% start with a clean slate
close all; clear all;clc;

% set the default colormap
set(groot,'DefaultFigureColormap',gray);
close 1;

%% ------------------------------------------------------------------------
%                 Artificial dataset path and settings
%%-------------------------------------------------------------------------

% path to the ThunderSTORM simulated stack of images in tiff format with 
% the accompanying ground truth positions in csv format
datapath = 'C:\Users\nanobio\Desktop\Elias\Deep-STORM Code';
tiff_filename = 'Artificialdataset.tif';
csv_filename = 'positions.csv';

% add the path to the data
addpath(genpath(datapath));

% upsampling factor for super-resolution reconstruction later on
upsampling_factor = 8;

% simulated camera pixel size in [nm]
camera_pixelsize = 100; 

% name of the training examples matfile for saving 
mat_filename = 'TrainingSet';

%% ------------------------------------------------------------------------
%              Extract patches and generate matching heatmaps
%%-------------------------------------------------------------------------

% gaussian kernel standard deviation [pixels]
gaussian_sigma = 1;

% training patch-size: needs to be dividable by 8 with no residual
patch_size = 26*upsampling_factor; % [pixels]

% number of patches to extract from each image
num_patches = 500; 

% maximal number of training examples
maxExamples = 10000;

% minimal number of emitters in each patch to avoid empty examples in case 
% of low-density conditions
minEmitters = 7;

% read the artificial acquisition stack
ImageStack = ReadStackFromTiff(tiff_filename);

% dimensions of acquired images
[M,N,numImages] = size(ImageStack);

% dimensions of the high-res grid
Mhr = upsampling_factor*M;
Nhr = upsampling_factor*N;

% create the high resolution grid with the appropriate pixel size
patch_size_hr = camera_pixelsize/upsampling_factor; % nm

% heatmap psf
psfHeatmap = fspecial('gauss',[7 7],gaussian_sigma);

% number of training patches in total
ntrain = min(numImages*num_patches,maxExamples); 

% initialize the training patches and labels
patches = zeros(patch_size,patch_size,ntrain);
heatmaps = zeros(patch_size,patch_size,ntrain);
spikes = false(patch_size,patch_size,ntrain);

% import all positions from ground truth csv
Activations = importdata(fullfile(datapath,csv_filename));
Data = Activations.data;
col_names = Activations.colheaders;

% check that user didn't take out columns when saving from ThunderSTORM
if length(col_names) < 8
    error('Number of columns in the ThunderSTORM csv file is less than 8!');
end 

% run over all frames and construct the training examples
k = 1;
skip_counter = 0;
f1 = figure('Name','Training set preparation');
for frmNum=1:numImages
    
    % cast tiff frame to double 
    y = double(ImageStack(:,:,frmNum));
    
    % upsample the frame by the upsampling_factor using a nearest neighbor
    yus = imresize(y,upsampling_factor,'box');  
    
    % read all the provided high-resolution locations for current frame
    DataFrame = Data(Data(:,2)==frmNum,:);
       
    % get the approximated locations according to the high-res grid pixel size
    Chr_emitters = max(min(round(DataFrame(:,3)/patch_size_hr),Nhr),1);
    Rhr_emitters = max(min(round(DataFrame(:,4)/patch_size_hr),Mhr),1);
    
    % find the linear indices of the GT emitters
    indEmitters = sub2ind([Mhr,Nhr],Rhr_emitters,Chr_emitters);
    
    % get the labels per frame in spikes and heatmaps
    SpikesImage = zeros(Mhr,Nhr);
    SpikesImage(indEmitters) = 1;
    HeatmapImage = conv2(SpikesImage,psfHeatmap,'same');
    
    % limit maximal number of training examples to 15k
    if k > ntrain
        break;
    else
        
        % choose randomly patch centers to take as training examples
        indxy = ClearFromBoundary([Mhr Nhr],ceil(patch_size/2),num_patches);
        [rp,cp] = ind2sub([Mhr,Nhr],indxy);

        % extract examples
        for i=1:length(rp)  

            % if a patch doesn't contain enough emitters then skip it
            if nnz(SpikesImage(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2))) < minEmitters
                skip_counter = skip_counter + 1;
                continue;
            else
                patches(:,:,k) = yus(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                    cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2));
                heatmaps(:,:,k) = HeatmapImage(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                    cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2));
                spikes(:,:,k) = SpikesImage(rp(i)-floor(patch_size/2)+1:rp(i)+floor(patch_size/2),...
                    cp(i)-floor(patch_size/2)+1:cp(i)+floor(patch_size/2));
                k = k + 1;
            end
        end
    end    
    
    % sanity check: show current low-res with spikes overlaid, and next to
    % it the high-res heatmap approximation
    figure(f1);
    subplot(1,2,1);imagesc(yus);hold on;plot(Chr_emitters,Rhr_emitters,'+r');axis off; axis square;
    if ~isempty(cp)
        hold on;rectangle('Position',[min(cp)-patch_size/2 min(rp)-patch_size/2 max(cp)-min(cp)+patch_size max(rp)-min(rp)+patch_size],'EdgeColor','b');
    end
    title('Low-res Measurements');
    subplot(1,2,2);imagesc(HeatmapImage);axis off; axis square;title(['Heatmap Approximation \sigma=' num2str(gaussian_sigma)]);  
    suptitle(['Extracting Training Examples: ' num2str(frmNum) ' out of ' num2str(numImages) ', Patches Acquired ' num2str(k)]);
    drawnow;
end

% check if the size of the training set is smaller than 5k to notify user
% to simulate more images using ThunderSTORM
if ((k-1) < 5000)
    warning('on');
    warning('Training set size is below 5K - Consider simulating more images in ThunderSTORM.')
end

% final resulting single patches dataset
patches = patches(:,:,1:k-1);
heatmaps = heatmaps(:,:,1:k-1);
spikes = spikes(:,:,1:k-1);

% save training dataset to a mat file
save(fullfile(datapath,mat_filename),'patches','heatmaps','spikes','-v7.3');
