
%all_training_sets = load('train_small.mat');
training_set = train_small{1};

training_labels = [];
training_features = [];

for i = 1: length(training_set)  
    
    % get all labels in our development training set
    training_labels = [training_labels, (training_set(i).labels)'];
    
    %get all the training features in our dataset
    for j = 1: length(training_set(i).images)
        
        % each image is a 28x28 array of pixels
        pixels = training_set(i).images(:,:,j);
        % for this naive approach, 
        % we will turn it into a row vector with
        % all the pixel values concatinated 
        row = reshape(pixels,1,[]);
        training_features = [training_features; row];
        
    end
end

lab = double(training_labels)';	
img = sparse(double(training_features));
model=train(lab, img, '-s 2');

%test = load('data/test.mat');
testing_set = test(1);

% use our model to make predictions about our training set
test_labels = testing_set.labels;
test_features = [];
for i = 1: length(testing_set.images)
    pixels = testing_set.images(:,:,i);
    row = reshape(pixels,1,[]);
    test_features = [test_features; row];
end
test_features = sparse(double(test_features));

prediction = predict(test_labels, test_features, model);