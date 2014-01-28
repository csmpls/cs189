%train = load('data/train_small.mat');
training_set = train{1};

labels = [];
trainingfeat = [];

for i = 1: length(training_set)  
    
    % get all labels in our development training set
    labels = [labels, (training_set(i).labels)'];
    
    %get all the training features in our dataset
    for j = 1: length(training_set(i).images)
        
        % each image is a 28x28 array of pixels
        pixels = training_set(i).images(:,:,j);
        % for this naive approach, 
        % we will turn it into a row vector with
        % all the pixel values concatinated 
        row = reshape(pixels,1,[]);
        trainingfeat = [trainingfeat; row];
        
    end
end

lab = double(labels)';	
img = sparse(double(trainingfeat));
model=train(lab, img, '-s 2')