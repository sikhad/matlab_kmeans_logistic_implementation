% Logistic Regression implementation

function [TestingAccuracy, TrainingAccuracy] = LogisticRegression()
    load('data.mat');

    % Initialize train, test, variables
    ntrain = size(TrainY, 1); 
    nfeat = size(TrainX, 2); 
    nclass = length(unique(TrainY)); 
    ytrain = sparse((1:ntrain), TrainY', ones(1,ntrain)); 
    [xtrain,mu,s] = zscore(TrainX); 
    xtest = zscore(TestX); 
    iterno = 1000; 

    % Iterate by computing loss function
    W = zeros(nfeat, nclass); 
    eta = 1e-5; 
    for i = 1:iterno    
        exp_wx_test = exp(xtest * W); 
        [~, idx] = max(exp_wx_test, [], 2); 

        exp_wx = exp(xtrain * W);
        g = zeros(nfeat, nclass); 
        for k = 1:nclass        
            g(:,k) = xtrain' * ytrain(:,k) ...
                - xtrain' * (exp_wx(:,k) ./ sum(exp_wx,2)); 
            W(:,k) = W(:,k) + eta * g(:,k); 
        end
    end
    
    % Use TrainX to generate TrainPredY  
    TrainPredY = zeros(size(TrainY, 1), 1);
    exp_wx_train = exp(xtrain * W); 
    [~, idx2] = max(exp_wx_train, [], 2); 
    TrainPredY = idx2;
    
    % Use TestX to generate TestPredY
    TestPredY = zeros(size(TestY, 1), 1);
    TestPredY = idx;

    % Calculate accuracies
    TrainingAccuracy = 100 * (1 - sum(TrainPredY ~= TrainY) / length(TrainY))
    TestingAccuracy = 100 * (1 - sum(TestPredY ~= TestY) / length(TestY))

end