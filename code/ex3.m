%train = load('data/train_small.mat');

% load 10000 training samples
set = train_small{7}; 

labels = [];
features = [];

for i = 1: length(set)  

    % get all labels in our development training set
    labels = [labels, (set(i).labels)'];

    %get all the training features in our dataset
    for j = 1: length(set(i).images)

        % each image is a 28x28 array of pixels
        pixels = set(i).images(:,:,j);
        % for this naive approach, 
        % we will turn it into a row vector with
        % all the pixel values concatinated 
        row = reshape(pixels,1,[]);
        features = [features; row];

    end
end

% perform cross-validation
fold_size = length(labels)/10;
for i = 0: 9
    
    i
    
    % set aside our test set
    fold_start = (i*fold_size)+1;
    fold_end = (i*fold_size)+fold_size;
    test_fold_labels = labels(fold_start:fold_end);
    test_fold_features = features(fold_start:fold_end,1:end);
    
    % the rest of the data goes into our testing set
    if (i == 0)
        training_fold_labels = labels(fold_end+1:end);
        training_fold_features = features(fold_end+1:end,1:end);
    elseif (i == 9)
        training_fold_labels = labels(1:fold_start-1);
        training_fold_features = features(1:fold_start-1,1:end);
    else
        t_fold_1 = labels(1:fold_start-1);
        t_fold_2 = labels(fold_end+1:end);
        training_fold_labels = cat(1, t_fold_1', t_fold_2');
        f_fold_1 = features(1:fold_start-1,1:end);
        f_fold_2 = features(fold_end+1:end,1:end);
        training_fold_features = cat(1, f_fold_1, f_fold_2);
    end
    
    % train
    lab = double(training_fold_labels)';
    img = sparse(double(training_fold_features));
    model = train(lab, img, '-s 2');
%     
%     testlab = test_fold_labels;
%     testfeat = sparse(double(test_fold_features));
%     prediction = predict(testlab, testfeat, model);
    
end


% 
% lab = double(training_labels)';	
% img = sparse(double(training_features));
% model=train(lab, img, '-s 2');
% 
