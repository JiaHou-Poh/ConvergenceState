function [] = MVPA_Convergence(vol_data,roi_list,roi_names,class,distance,repl,partition, par_vect,num_cluster,save_fldr)
%%
% Examines state representation in each ROI by computing distance of each
% item with a cluster centroid.
%
% Convergence is defined as the distance of each point to the
% cluster centroid.
%
% Also computes item-wise centrality, defined as the distance of each point
% to every other point belonging to the same class.
%
% Input:
% 1) vol:
%   Cell array of 3D .nii or .nii.gz file. For array of 3D files each
%   volume should occupy a separate row.
%
% 2)roi_list:
%   Cell array of binary ROI masks (.nii or .nii.gz).
%
% 3)roi_names:
%   Cell array of string to save the ROIs ask.
%
% 4)class:
%   Vector with length equal to vol. A vector of 1s can be used if no class
%   differentiation is required.
%
% 5)distance:
%   Distance measure to use. Refer to pdist2 for documentation on available
%   distance metrics. Commonly used distances includes
%   'correlation,'cosine' & 'euclidean'.
%
% 6)repl:
%   Number of repetition to run the K-means algorithm with different
%   starting points.
%
% 7) partition:
%   If partition == 1, centroid will be defined using P-1 partition and
%   distance to centroid will be measured using the left out partition.
%   Point-to-point distance will also exclude trials from the same
%   partition. Requires input parvect
%
% 8) par_vect:
%   If partition == 1, parvect will be a vector of values corresponding to
%   the partition that each item belongs to.
%
% 9) num_cluser:
%   Number of cluster (k) for k-means clustering.
%
% 10) save_fldr:
%       Directory for saving the outputs.
%
% Note: This function requires MRIread from Freesurfer.
%
% Edit 1/9/2020
% Added mean activation to variables saved.
%%
rep_dist = struct();
tmp_data = vol_data{1};
tmp_vol = MRIread(tmp_data);
brainvx = size(tmp_vol.vol,1) * size(tmp_vol.vol, 2) * size(tmp_vol.vol, 3);

num_roi = size(roi_list);
num_item = size(vol_data,1);

class_ls = unique(class);
class_ls = class_ls(~isnan(class_ls));
num_cls = length(class_ls);

if partition == 1
    par_ls = unique(par_vect);
else
    par_ls = 1;
    %par_vect = ones(num_item,1);
end
num_par = length(par_ls);

p2cs = NaN(num_item,num_cluster);
short_p2cs = NaN(num_item,1);

p2ps = NaN(num_item,1);
class_p2ps = NaN(num_item,num_cls);
within_p2ps = NaN(num_item,1);
between_p2ps = NaN(num_item,1);

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
        
        %% Point to point distances
        % Across all points
        tmp_p2p = pdist2(train_data,test_data,distance);
        p2ps(test_idx,1) = transpose(mean(tmp_p2p,1));
        
        % Within-class and between-class point to point
        for c = 1 : num_cls
            test_cls = class_ls(c);
            
            t_idx = find(class == test_cls);
            c_test_idx = intersect(t_idx,test_idx);
            c_test_data = roi_data(c_test_idx,:);
            
            for d = 1 : num_cls
                train_cls = class_ls(c);
                
                nt_idx = find(class == train_cls);
                c_train_idx = intersect(nt_idx,train_idx);
                c_train_data = roi_data(c_train_idx,:);
                
                tmp_cls_p2p = pdist2(c_train_data,c_test_data,distance);
                class_p2ps(c_test_idx,d) = transpose(mean(tmp_cls_p2p,1));
            end
            within_p2ps(c_test_idx,1) = class_p2ps(c_test_idx,c);
            if num_cls > 1
                nc = setdiff(class_ls,c);
                between_p2ps(c_test_idx,1) = transpose(mean(class_p2ps(c_test_idx,nc),1));
            else
                between_p2ps(c_test_idx,1) = NaN;
            end
        end
    end
    % Assign to structure
    rep_dist.mean_act = roi_act;
    rep_dist.point2centroids.p2cs = p2cs;
    rep_dist.point2centroids.p2cs_shortest = short_p2cs;
    
    rep_dist.point2points.p2ps = p2ps;
    rep_dist.point2points.within_class_p2ps = within_p2ps;
    rep_dist.point2points.between_class_p2ps = between_p2ps;
    
    % Saving
    savefname = [save_fldr wroi '_' num2str(num_cls) 'cls_k' num2str(num_cluster) '_' distance '_RepresentationalDistance.mat'];
    save(savefname,'rep_dist','centroid_arr','vol_data','roi_list','roi_names',...
        'class','distance','repl','partition','par_vect','save_fldr');
end
