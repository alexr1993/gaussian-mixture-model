function im = getImage( file )

% read in images
im = imread( file );
%im = rgb2gray(im);
im = double(im)/255;
im=ceil(im);
%make sure it is actually binary
im=ceil(im);

%make in to an outline
im=bwmorph(im,'remove');
im=bwmorph(im,'thin');
im=bwmorph(im,'spur');