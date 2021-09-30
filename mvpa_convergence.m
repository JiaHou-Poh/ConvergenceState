function [] = mvpa_convergence(vol_data,roi_list,roi_names,distance,repl,partition, par_vect,num_cluster,save_fldr)
%%
% Examines state representation in each ROI by computing distance of each
% item with a cluster centroid.
%
% Convergence is defined as the distance of each point to a
% cluster centroid. Cluster number can be adjusted based on the number of
% expected 'states'.
%
% Also computes item-wise representational distance, defined as the distance of each point
% to every other point. This measure can also take into account class
% membership, computing a separate distance for within- and between- class
% distance.
%
% Input:
% 1) vol:
%   Cell array of 3D .nii or .nii.gz file. For array of 3D files each
%   volume should occupy a separate row.
%   !!IMPORTANT!! - ROI mask and the brain data (vol) must have the same
%   dimensions and must be aligned in the same space.
%
% 2)roi_list:
%   Cell array of binary ROI masks (.nii or .nii.gz). 
%   !!IMPORTANT!! - ROI mask and the brain data (vol) must have the same
%   dimensions and must be aligned in the same space.
%
% 3)roi_names:
%   Cell array of string to save the ROIs as.
%
% 4)distance:
%   Distance measure to use. Refer to pdist2 for documentation on available
%   distance metrics. Commonly used distances includes
%   'correlation,'cosine' & 'euclidean'.
%
% 5)repl:
%   Number of repetition to run the K-means algorithm with different
%   starting points. Note that when K = 1, solution should be deterministic
%   regardless of the distance metric used.
%
% 6) partition:
%   If partition == 1, centroid will be defined using P-1 partition and
%   distance to centroid will be measured using the left out partition.
%   Point-to-point distance will also exclude trials from the same
%   partition. Requires input par_vect with values ranging from 1 to P.
%
% 7) par_vect:
%   If partition == 1, par_vect will be a vector of values from 1 to P corresponding to
%   the partition that each item belongs to (e.g. run or block number).
%   If partition ==0, input a vector of ones.
%
% 8) num_cluser:
%   Number of cluster (k) for k-means clustering.
%
% 9) save_fldr:
%       Directory for saving the outputs.
%
% Outputs a single structure 'rep_dist' with the following fields:
% 1) mean_act: 
%       mean value over the entire ROI.
% 
% 2) point2centroids.p2cs:
%       distance of each point to the respective k-centroids.
%
% 3) point2centroids.p2cs_shortest:
%       distance of each point to the centroid which it is closest to.
%
% Note: This function requires MRIread from Freesurfer.
%
% Edit 1/9/2020
% Added mean activation to variables saved.
%
% Edit 15/7/2021
% Added comments for ouput structure and commented out the declaration of
% par_vect in line 100 (should take from input).
%
% Edit 29/7/2021
% Remove analysis for point to point distance to streamline the function.
% Point to point distance will be released as a separate function.
% If anyone is using the point to point analysis from prior releases,
% please note that it has not been thoroughly tested.
% Also remove the input required for 'class'
%
% Edit 29/9/2021
% Added comments to indicate that par_vect should be a vector of 1s when
% partition == 0.
%%
rep_dist = struct();
tmp_data = vol_data{1};
tmp_vol = MRIread(tmp_data);
brainvx = size(tmp_vol.vol,1) * size(tmp_vol.vol, 2) * size(tmp_vol.vol, 3);

num_roi = size(roi_list);
num_item = size(vol_data,1);

% class_ls = unique(class);
% class_ls = class_ls(~isnan(class_ls));
% num_cls = length(class_ls);

if partition == 1
    par_ls = unique(par_vect);
else
    par_ls = 1;
    %par_vect = ones(num_item,1);
end
num_par = length(par_ls);

p2cs = NaN(num_item,num_cluster);
short_p2cs = NaN(num_item,1);

% p2ps = NaN(num_item,1);
% class_p2ps = NaN(num_item,num_cls);
% within_p2ps = NaN(num_item,1);
% between_p2ps = NaN(num_item,1);

centroid_arr = cell(num_par,1);

% Load data
full_data = NaN(num_item,brainvx);
for i = 1 : size(vol_data,1)
    tmp_data = vol_data{i};
    tmp_vol = MRIread(tmp_data);
    brainvx = size(tmp_vol.vol,1) * size(tmp_vol.vol, 2) * size(tmp_vol.vol, 3);
    
    full_data(i,:) = transpose(reshape(tmp_vol.vol,brainvx,1));
end

for s = 1 : num_roi
    roi_data = roi_list{s};
    wroi = roi_names{s};
    fprintf('Starting analysis for ROI - %s \n', wroi);
    % Load ROI
    roi = MRIread(roi_data);
    roi_idx = find(roi.vol==1);
    
    roi_data = full_data(:,roi_idx);
    roi_act = nanmean(roi_data,2);
    
    for i = 1 : num_par
        if partition == 1
            test_par = par_ls(i);
            train_par = par_ls(par_ls~=test_par);
        elseif partition == 0
            test_par = 1;
            train_par = 1;
        end
        
        train_idx = find(ismember(par_vect,train_par)==1);
        test_idx = find(par_vect== test_par);
        
        train_data = roi_data(train_idx,:);
        test_data = roi_data(test_idx,:);
        
        %% Distance to centroid
        % Compute centroid and each point's distance to centroid using K-means
        % clustering
        if contains (distance,'euclidean')
            clustdist = 'sqeuclidean';
        else
            clustdist = distance;
        end
      
        kmeans_out = struct();
        kmeans_out.dist = clustdist;
        [kmeans_out.train_cluster,kmeans_out.centroids,kmeans_out.within_train_dist,kmeans_out.all_train_dist] = ...
            kmeans(train_data,num_cluster,'Distance',clustdist,'Replicates',repl);
        
        centroid_arr{i,1} = kmeans_out;
        
        % Apply test data to centroid
        centroids = kmeans_out.centroids;
        tmp_p2c = pdist2(centroids,test_data,distance);
        p2cs(test_idx,:) = tmp_p2c';
        
        [shortest_p2c,test_cluster] = pdist2(centroids,test_data,distance,'Smallest',1);
        short_p2cs(test_idx,1) = shortest_p2c';
    end
    % Assign to structure
    rep_dist.mean_act = roi_act;
    rep_dist.point2centroids.p2cs = p2cs;
    rep_dist.point2centroids.p2cs_shortest = short_p2cs;
    
    % Saving
    savefname = [save_fldr wroi '_k' num2str(num_cluster) '_' distance '_RepresentationalConvergence.mat'];
    save(savefname,'rep_dist','centroid_arr','vol_data','roi_list','roi_names',...
        'distance','repl','partition','par_vect','save_fldr');
end
