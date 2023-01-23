%---- Reading the dataset ----
bank_data = readtable('bank_full.xlsx');
%-- Rearranging the columns keeping numeric columns first followed by categorical columns
bank_data = bank_data(:,[1 6 10 12 13 14 15 2 3 4 5 7 8 9 11 16 17]);
%label encoding for the response variable (1, if bank_Train.y = 'yes', or 0 if bank_Train.y = 'no')
bank_data.y = strcmp(bank_data.y,"yes");
% --- validating data by applying holdout of 35%
pt = cvpartition(bank_data.y,'holdout',0.35);
%--- dividing the data into the testset and the train set -----
bank_Train = bank_data(training(pt),:);
bank_Test = bank_data(test(pt),:);
%----------------Model A - Decision Tree -----------------
%{
There are too many predictors which are not relevant . So selecting relevant features and only 
passing only those into features into the decision tree. Using matlab build
in function to calculate the importance of each predictor and then store it
in a variable named 'p'.
base model named 'treeMdl' is created which is then used to predict the importance of each
predictors
%}

treeMdl= fitctree(bank_Train,'y','SplitCriterion', 'gdi','MaxNumSplits', 100);
p = predictorImportance(treeMdl);
%visualize p using bar
columnNames = categorical(bank_data.Properties.VariableNames(1:end-1));
figure('Name','Predictor Importance graph')
bar(columnNames,p)
ylabel("Predictor Importance");
title("Importance of each Predictor on Response");

%{
Keeping only those predictors that have value greater than 0.006.
Creating a logical vector named 'toKeep' whose values are true if the corresponding values
of p are greater than 0.006 and false otherwise
%}
toKeep = p >0.00006;

%{
using 'toKeep' variable to create a new table named bankPartTrain (train set) and bankPartTest
 (test set) which contains only the predictors with p greater than 0.006 and the response variable y.
%}
bankPartTrain = bank_Train(:,[toKeep true]);
bankPartTest = bank_Test(:,[toKeep true]);

%{
Creating a multi-class model 'fitceoc'. It returns a full, trained,
multiclass, error-correcting output codes (ECOC) model using the predictors
in table .
Using Hyper parameter optimization with fitceoc method .
Hyper parameter with tree tempalate searches for minimum leaf size , split
criteria,maximum number of splits .
Seeds of the random number generators are set using rng and tallrng
%}
rng('default') %for reproducibility
tallrng('default')

% tree template is set for decision tree 
t = templateTree();

%tic-toc is used to calculate evaluation time
tic; 

%{
'MinLeafSize' value that minimizes holdout cross-validation loss.
(Specifying 'auto' uses 'MinLeafSize'for tree template).

Binning ('NumBins',50) — When we have a large training data set, we can speed up training 
(a potential decrease in accuracy) by using the 'NumBins' name-value pair argument. This argument
 is valid only when fitcecoc uses a tree learner. If we specify the 'NumBins' value, then the software
 bins every numeric predictor into a specified number of equiprobable bins, and then grows trees on the 
bin indices instead of the original data. Numbin value is taken as 50 on
basis of accuracy .

Parallel computing ('Options',statset('UseParallel',true)) — With a Parallel Computing Toolbox license, 
you can speed up the computation by using parallel computing, which sends each binary learner to a worker
 in the pool. The number of workers depends on your system configuration. When you use decision trees for 
binary learners, fitcecoc parallelizes training '.

onevsall takes K number of binary learners.That is For each binary learner, one class is positive
and the rest are negative. 
This design exhausts all combinations of positive class assignments.
%}

options = statset('UseParallel',true);
treeCecocMdl = fitcecoc(bankPartTrain,'y','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
struct('AcquisitionFunctionName','expected-improvement-plus'),'Learners',t,...
'NumBins',52,'Options',options);
toc;

%calculating the loss with training and test set
errTreeEoc = resubLoss(treeCecocMdl); %with trainset
errTreeTstEoc = loss(treeCecocMdl,bankPartTest); %with testing

%{
Calling function 'fnAucRocValue' for calculating auc ,threshold ,roc values and predicting
labels. 
%}
[XEoc,YEoc,TEoc,AucEoc,prdctEoc] = fnAucRocValue(treeCecocMdl,bankPartTrain);  %with training set
[XEocTst,YEocTst,TEocTst,AucEocTst,prdctEocTst] = fnAucRocValue(treeCecocMdl,bankPartTest); %with testing set

%{
Evaluating model by checking for accuracy,precision,recall,Fscore and
missclassification rate.
Function named 'fnAcc' is called to calculate above values.
%}
[accuracyEoc,missClassRateEocT,PrecisionEocT,RecallEocT,FscoreEocT] = fnAcc(prdctEoc,bankPartTrain.y); %for train data
[accuracyEocTst,missClassRateEocTst,PrecisionEocTst,RecallEocTst,FscoreEocTst] = fnAcc(prdctEocTst,bankPartTest.y); %for train data

%{
Plotting the above calculated values on graph .
%}

%plot for Roc
figure('Name','ROC Comparision Curves')
plot(XEocTst,YEocTst) %test data
hold on
plot(XEoc,YEoc) %train data
legend("Roc for Test set","Roc for Train set")
xlabel('False positive rate') ;
ylabel('True positive rate');
title('ROC Curves for Decision Tree')
hold off

%{
plot for accuracy,precision,Fscore,recall and missclassified rate with test set
%}
Values = [accuracyEocTst,PrecisionEocTst,RecallEocTst,FscoreEocTst,missClassRateEocTst];
Performance_parameter = categorical(["Accuracy","Precision","Recall","Fscore","Missclassified rate"]);
figure('Name','Performance parameter plot')
bar(Performance_parameter,Values)
title("Performance Parameters Plot for Decision Tree");

%confusion matrix for the final optimized model with test set
figure('Name','Confusion Matrix for Decision Tree')
confusionchart(bankPartTest.y,prdctEocTst)
title("Performance Parameters Plot for Decision Tree");

% ----- function for calculating accuracy,precision,recall,Fscore and missclassification
% rate 

function [accuracy,missClassRate,Precision,Recall,Fscore] = fnAcc(predictions,actualLabels)
isCorrect = predictions==actualLabels; %labels predicted correctly
isWrong = predictions~=actualLabels; %labels not predicted correctly
accuracy = sum(isCorrect)/numel(predictions);
missClassRate = sum(isWrong)/numel(predictions); %missclassified rate

%calculating confusion matrix
[confMat,order] = confusionmat(actualLabels,predictions);
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(i,:));
end
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));
end
Precision=sum(precision)/size(confMat,1);
Recall=sum(recall)/size(confMat,1);
Fscore = (2*Recall*Precision/(Recall+Precision));
end

%--- function to calculate auc values,threshold values,scores and prediction values
function [X,Y,T,Auc,label] = fnAucRocValue(mdl,data)
%Predicting response for observations 
[label,score] = predict(mdl,data);
% Computing auc and threshold 
[X,Y,T,Auc] = perfcurve(data.y,score(:,mdl.ClassNames),'true');%perfcurve stores the threshold values in the array T.
end