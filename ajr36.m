%% ajr36 cm20220 classifier coursework
%

% more data available at www.lems.brown.edu/vision/researchareas/siid
%
% march 2013
%
% step 1: generate chain code
%
% step 2: fourier transform discard everything outside the interval (0,
% ~20)
%
% step 3: Construct a Gaussian mixture model (fully supervised)
%
% step 4: Use Bayes theorem to classify data based on the conditional
% probability density function obtained from the Gaussian
%
% step 5: Create confusion matrix which indicates the quality of the
% classifier
%

%% Optional cool stuff
%
% - Create my own images and try to mess up my classifier with them, e.g.
% using images that shouldn't belong to any class
%
% - Try different feature vectors, what effect does N have
%
% - Try fitting GMM directly
%
% - Try a different classifier all together, e.g. K-means
%
% - KL-divergence (this is a rigour measure amongsth others)
%

%% Assessment

%-----------------+--------------------+----------------------------------%
% class boundary  |  coding milestone  |  testing rigour                  %
%-----------------+--------------------+----------------------------------%
%     3rd         |  Feature vectors   | Eyeball distribution             %
%                 |                    |                                  %
%     2.2         |  Eigenmodels       | Are groups gaussian distributed? %
%                 |                    |                                  %
%     2.1         |  GMM               | Confusion matrix                 %
%                 |                    |                                  %
%     1st         |  Optimised         | Confusion vs feature dimension   %
%                 |  classifier        |                                  %
%-----------------+--------------------+----------------------------------%

% - Hand in a good report
%
% - Comment code well
%
%

%% Clear all vars and close all figures
clear all;
close all;

%% input libraries

% possibly split them into classes for supervised training
categories{1} = 'aliens/';
categories{2} = 'butterflies/';
categories{3} = 'faces/';
categories{4} = 'stars/';

trainingsetsizes = zeros(4,1);

confvsdim = [];

% 5:5 will test just 5 dimensions, 5:6 will test 5 dimensions
% then 6 dimensions etc
for Dim = 5:5
    
featurespace = cell(4, 1); % each cell contains N feature vectors (20Dims)
%Dim = 20; % dimensions of feature vector
mus = zeros(4, Dim); % (Components, Dimensions)
sigmas = zeros(Dim, Dim, 4); % (Dimensions, Dimensions, Components)

% for all 4 classes
% read in images, convert to chain code, apply fourier transform,
% restrict dimensions, calculate means and covariances
for class = 1:4
    lib = ['trainingset/' categories{class}]; % training set directory
    % error handling
    if exist(lib, 'dir') == 0
        disp('Error: Image directory set incorrectly');
        return;
    end

    %% create chain code for training set
    %
    % chain code is a series of 3D vectors, dimensions are x, y and direction
    % The direction is a 0-7 value, like so:
    %
    %          5 6 7  
    %          4 . 0
    %          3 2 1
    %

    % read in training set
    images = dir([lib '*.gif']);

    % find size of training set
    trainingsetsizes(class) = length(images);

    % allocate chain code array
    chains = cell(1,trainingsetsizes(class));

    % convert each image in the training set to chain code
    for i = 1:trainingsetsizes(class)
        
        % read in image
        image = getImage([lib images(i).name]);
        
        % convert to chain code - chaincode function courtesy of pmh
        chains{i} = chainCode(image);
    end
    
    %% plot some chain code to check if it's loaded properly
    %{
    c = chains{6};

    figure;
    hold on;
    plot(c(1,:), c(2,:), 'r.');
    %}

    %% Create feature vectors using fft

    % allocate array for transformed chain code (feature vectors)
    featurevectors = cell(1,trainingsetsizes(class));

    % for each chain code, apply fft
    for i = 1:trainingsetsizes(class)
        
        % access 3D array from cell in chains
        % actual chain code is the third row (3,:)
        chain = chains{i}(3,:); 

        % fft is needed because chain code feature vectors cannot be used
        % to distinguish shapes of different sizes and high frequency data
        % will vary too much
        freqs = fft(chain); % fourier transform chain code

        % select information from 2 - halfway (nyquist)?
        nyquistrange = freqs(2:ceil(size(freqs,2)/2));

        featurevectors{i} = abs(nyquistrange(1:Dim))'; %fea vector is 1:D components

    end

    % convert cell array to matrix
    % columns are feature vectors i.e.F D x #imgs matrix
    featurespace{class} = cell2mat(featurevectors);

    %% Perform PCA
    %{
    allpoints = [featurespace{1} featurespace{2} featurespace{3} featurespace{4}];
    
    all_cov = cov(allpoints');
    all_mean = mymean(allpoints);
    [U,L] = eig(all_cov); % evd

    L = diag(L); % evalues
    [L,i] = sort(L, 'descend'); % sort evalues
    U=U(:,i); % sort evectors the same way 

    energy = cumsum(L) / sum(L);
    energy = energy < 0.99;

    U = U(:, energy);
    L = L(energy);

    
    if class == 4
        %project
    
        %use this line on input test data
        all_points_best = U'*(allpoints - repmat(all_mean,1,size(allpoints,2)));
        featurespace{1} = all_points_best(:,1:42);
        featurespace{2} = all_points_best(:,1:50);
        featurespace{3} = all_points_best(:,1:100);
        featurespace{4} = all_points_best(:,1:28);
    
    end
    %}
    %% create eigenmodels
    
    % K = no. components
    % D = dimensions of feature vector
    % 
    % GMM = gmdistribution(mu = K x D, sigma = D x D x K)
    
    % mu is the mean
    % sigma is the covariance
    
    % calculate covariance of set resulting in D x D matrix
    sigmas(:,:,class) = cov(featurespace{class}'); % recreate
    
    % calculate mean of class (1 x D matrix)
    mus(class,:) = mymean(featurespace{class})'; % recreate
    
    % pmh suggested it would be good if i made my own mean and covariance fns
    
end


%% Create GMM

% GMM = gmdistribution(mu = K x D, sigma = D x D x K)
GMM = gmdistribution(mus, sigmas);


%% Calculate Priors

% divide each quantity by total
priors = trainingsetsizes ./ sum(trainingsetsizes);


%% Analyse effectiveness of GMM using confusion matrix

conf = zeros(4,4);

% read in test set
% --> chain code --> feature vectors

testsetsizes = zeros(4,1);
testsetspace = cell(4, 1); % each cell contains N feature vectors (20Dims)

% for all 4 classes
% read in images, convert to chain code, apply fourier transform,
% restrict dimensions, calculate means and covariances
for class = 1:4
    lib = ['testset/' categories{class}]; % training set directory
    % error handling
    if exist(lib, 'dir') == 0
        disp('Error: Image directory set incorrectly');
        return;
    end

    %% create chain code for test set

    % read in test set
    images = dir([lib '*.gif']);

    % find size of training set
    testsetsizes(class) = length(images);

    % allocate chain code array
    chains = cell(1,testsetsizes(class));

    % convert each image in the training set to chain code
    for i = 1:testsetsizes(class)
        
        % read in image
        image = getImage([lib images(i).name]);
        
        % convert to chain code - chaincode function courtesy of pmh
        chains{i} = chainCode(image);
    end
    
    %% plot some chain code to check if it's loaded properly
    %{
    c = chains{6};

    figure;
    hold on;
    plot(c(1,:), c(2,:), 'r.');
    %}

    %% Create feature vectors using fft

    % allocate array for transformed chain code (feature vectors)
    featurevectors = cell(1,testsetsizes(class));

    % for each chain code, apply fft
    for i = 1:testsetsizes(class)
        
        % access 3D array from cell in chains
        % actual chain code is the third row (3,:)
        chain = chains{i}(3,:); 

        % fft is needed because chain code feature vectors cannot be used
        % to distinguish shapes of different sizes and high frequency data
        % will vary too much
        freqs = fft(chain); % fourier transform chain code

        % select information from 2 - halfway (nyquist)?
        nyquistrange = freqs(2:ceil(size(freqs,2)/2));

        featurevectors{i} = abs(nyquistrange(1:Dim))'; %fea vector is 1:D components

    end

    % convert cell array to matrix
    % columns are feature vectors i.e.F D x #imgs matrix
    testsetspace{class} = cell2mat(featurevectors);
    
    for i = 1:testsetsizes(class)

    %% Classifier is In classify.m

    % probability of feature belonging to class is prior * N(x|mu, c)
    % probability of x belonging to eigenmodel is
    % exp( -0.5* mahalanobis distance ) / sqrt( prod(Cov) * (2*pi)^(number of dimensions) );
    x = testsetspace{class}(:,i); %sample input

    result = classify(x, GMM, priors); % i made dis

    % If an img is in class 1 then put it in row 1
    % If an img is classified as class 1 then put it in column 1
    conf(class, result) = conf(class, result) + 1;




    end
end

    %% Optimise classifier by checking the confusion matrix against dimensions
    % of the feature vector
    percentage = percentagesuccess(conf);
    
    confvsdim = [confvsdim; Dim, percentage]    

end
%% K-Means


%% Naive Bayes?

