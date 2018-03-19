function ImageStack = ReadStackFromTiff(filename)
% ImageStack = ReadStackFromTiff(filename)
% this function reads the image stack from the tiff file provided by
% filename.
%
% Inputs
% filename      -   tif stack filename (char)
%
% Outputs
% ImageStack    -   the resulting 3D image stack matrix (uint16)
%
% Written by Elias Nehme, 25/08/2017

    % get image info and number of images
    InfoImage=imfinfo(filename);
    mImage=InfoImage(1).Width;
    nImage=InfoImage(1).Height;
    NumberImages=length(InfoImage);
    
    % read images using Tiff object
    ImageStack=zeros(nImage,mImage,NumberImages,'uint16');
    TifLink = Tiff(filename, 'r');
    for i=1:NumberImages
       TifLink.setDirectory(i);
       ImageStack(:,:,i)=TifLink.read();
    end
    TifLink.close();

end

