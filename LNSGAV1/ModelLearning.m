function mlp = ModelLearning(Problem, inputs, targets)
% Training the MLP model

    LoserDec  = inputs.decs;
    WinnerDec = targets.decs;

    Lower = Problem.lower;
    Upper = Problem.upper;

    LoserDec = (LoserDec-repmat(Lower,size(LoserDec,1),1))./repmat(Upper-Lower,size(LoserDec,1),1);
    WinnerDec = (WinnerDec-repmat(Lower,size(WinnerDec,1),1))./repmat(Upper-Lower,size(WinnerDec,1),1);

    num_D     = size(LoserDec,2);
    epoch = 1;
    %epoch = ceil(num_D./1000)*1;
    %num_L = max(ceil(num_D./10),10);
    num_L = 5;
    %num_L = ceil(num_D./1000)*10;

    %% Train MLP
    mlp = SMLP(num_D,num_D,1,num_L,0.1,0.9);
    for i = 1:epoch
        mlp.train(LoserDec, WinnerDec);
    end 
end