sounds = ["Rain", "Waves", "Fire", "Crickets", "Birds"];
[trainInd,valInd,testInd] = divideind(36,1:26,27:31,31:36);
trainSize = (length(trainInd) + length(valInd)) * 5;
Xtrainval = []; 
Ytrainval = [];
Xtrain = [];
Ytrain = [];
Xval = [];
Yval = [];
Xtest = [];
Ytest = [];
valscores = [];
testscores = [];
ix = zeros(trainSize,1);

tStart = tic;
for s = 1:length(sounds)
    Ytrain = [Ytrain; ones(length(trainInd),1) * (s - 1)];
    Yval = [Yval; ones(length(valInd),1) * (s - 1)];
    Ytest = [Ytest; ones(length(testInd),1) * (s - 1)];
    for i = 1:length(trainInd)
        [xx,fs] = audioread(strcat(sounds(s), "_", int2str(trainInd(i)), ".wav"));
        c = envclass(xx,fs);
        Xtrain = [Xtrain; c];
    end
    for i = 1:length(valInd)
        [xx,fs] = audioread(strcat(sounds(s), "_", int2str(valInd(i)), ".wav"));
        c = envclass(xx,fs);
        Xval = [Xval; c];
    end
    for i = 1:length(testInd)
        [xx,fs] = audioread(strcat(sounds(s), "_", int2str(testInd(i)), ".wav"));
        c = envclass(xx,fs);
        Xtest = [Xtest; c];
    end
end

for k = 1:20
    valscore = 0;
    Ypredict = KNN(k,Xtrain,Ytrain,Xval,'Euclidian'); %[predicted_labels,nn_index,accuracy] = KNN_(3,training,training_labels,testing,distmethod)
    for i = 1:length(Ypredict)
        if Ypredict(i) == Yval(i)
            valscore = valscore + 1;
        end
    end
    valscore = double(valscore / length(Ypredict)) * 100.0;
    valscores = [valscores valscore];
end

for n = 1:1000
    ix = randperm(trainSize);
    Xtrainval = [Xtrain; Xval];
    Ytrainval = [Ytrain; Yval];
    Xtrainval = Xtrainval(ix,:);
    Ytrainval = Ytrainval(ix,:);
    Xtrain = Xtrainval(1:130,:);
    Ytrain = Ytrainval(1:130,:);
    Xval = Xtrainval(131:155,:);
    Yval = Ytrainval(131:155,:);
    for k = 1:20
        valscore = 0;
        Ypredict = KNN(k,Xtrain,Ytrain,Xval,'Euclidian'); %[predicted_labels,nn_index,accuracy] = KNN_(3,training,training_labels,testing,distmethod)
        for i = 1:length(Ypredict)
            if Ypredict(i) == Yval(i)
                valscore = valscore + 1;
            end
        end
        valscore = double(valscore / length(Ypredict)) * 100.0;
        valscores(k) = valscores(k) + valscore;
    end
end
valscores = valscores / 1000;
[valscore,pos] = max(valscores);
disp('Validation k = ')
disp(pos)
disp('Validation Accuracy = ')
disp(valscore)
Xtrainval = [Xtrain; Xval];
Ytrainval = [Ytrain; Yval];
for k = 1:20
    testscore = 0;
    Ypredict = KNN(k,Xtrainval,Ytrainval,Xtest,'Euclidian');
    for i = 1:length(Ypredict)
        if Ypredict(i) == Ytest(i)
            testscore = testscore + 1;
        end
    end
    testscore = testscore / length(Ypredict) * 100.0;
    testscores = [testscores testscore];
end
[testscore,pos] = max(testscores);
disp('Test k = ')
disp(pos)
disp('Test Accuracy = ')
disp(testscore)
tElapsed = toc(tStart)
save('vars','Xtrainval','Ytrainval')