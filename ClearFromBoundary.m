function [indxy,indxy_cfb] = ClearFromBoundary(dim,margin,Nemitters)
% [indxy,indxy_cfb] = ClearFromBoundary(dim,margin,Nemitters)
% function picks indices that are margin pixels apart from the boundary to 
% prevent patch truncation in the resulting dataset.
%
% Inputs
% dim           -   the dimensions of the image
% margin        -   the desired margin pixels
% Nemitters     -   number of emitters to sample
%
% Outputs
% indxy         -   the chosen valid indices
% indxy_cfb     -   all valid locations
%
% Written by Elias Nehme, 25/08/2017

    % dimensions of the image
    M = dim(1);
    N = dim(2);
    
    % create index grid 
    [rows,cols] = meshgrid(1:M,1:N);
    
    % valid rows and columns
    rows_cb = rows( rows>=margin & cols>=margin & (rows<=(M-margin)) & (cols<=(N-margin)));
    cols_cb = cols( rows>=margin & cols>=margin & (rows<=(M-margin)) & (cols<=(N-margin)));
    
    % appropriate valid indices
    indxy_cfb = sub2ind([M,N],rows_cb,cols_cb);
    
    % chose "random" x-y locations from the valid set
    indxy = randsample(indxy_cfb,Nemitters);
end

