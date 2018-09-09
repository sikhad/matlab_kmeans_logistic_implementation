% K-means implementation
% Input data: pixels (R, G, B values), K clusters
% Output: cluster assigment, centroid location

function [ class, centroid ] = L1kmeans( pixels, K )

    % randomly initialize centroid with data points;
    centroid = pixels(randsample(size(pixels,1),K),:);
    c_old = centroid + 10;

    while (norm(centroid - c_old, 'fro') > 1e-6)
        
        % record previous centroid;
        c_old = centroid;
     
        % calculate the L1 norm;
        diff = [];

        for i = 1:K   
            diff = cat(2, diff, sum(abs(bsxfun(@minus, pixels, centroid(i,:))),2));
        end
        [~, class] = min(diff, [], 2);

        % recompute centroids;
        newcentroid = [];
        for j = 1:K
            % find indices of class j; 
            ind = find(class(:,1)==j);
            % find median;
            newcentroid = cat(1, newcentroid, median(pixels(ind(:),:,:),1));
        end

        centroid = newcentroid;
             
end