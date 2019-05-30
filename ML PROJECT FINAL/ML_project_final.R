rm(list = ls())
setwd("C:\\Users\\karanbari\\Documents\\data science\\ML\\ML FINAL\\FINAL CODE")
data = read.csv("bank-additional-full.csv")
library(caTools) #sampling the dataset
library(ROSE) ## for caluclating and plotting the evaluation
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(randomForest)
library(DMwR)
library(xgboost)
library(data.table)
library(e1071)
library(mlr)
library(corrgram)
library(ggplot2)
library(FSelector) #this will require java to be installed inn your machine
############################################ data cleaning
View(data)
#### correlation analysis for numerical variables
a = c(1,11,12,13,14,16,17,18,19,20)
numerical= data[ ,a]
View(cor(numerical))

#checking correlation for all variables
new_tr <- model.matrix(~.+0,data = data) ## one hot encoding for all the categorical value
View(new_tr)

df2 = cor(new_tr)
hc = findCorrelation(abs(df2), cutoff=0.6) # put any value as a "cutoff" 
hc = sort(hc)
reduced_Data = df2[,(hc)]
View(reduced_Data)

##### replacing unknown values in the categorical variables
data$job = as.character(data$job)
data$marital = as.character(data$marital)
data$education = as.character(data$education)
data$housing = as.character(data$housing)

# for job replacing Unknownwith Admin(max frequency)
for(i in 1:length(data$job)){
  if(data$job[i] == "unknown"){
    data$job[i] = "admin."
  }
}

# for marital replacing Unknown with married(max frequency)
for(i in 1:length(data$marital)){
  if(data$marital[i] == "unknown"){
    data$marital[i] = "married"
  }
}

# for education replacing Unknown with university.degree(max frequency)
for(i in 1:length(data$education)){
  if(data$education[i] == "unknown"){
    data$education[i] = "university.degree"
  }
}

# for housing replacing Unknown with yes(max frequency)
for(i in 1:length(data$housing)){
  if(data$housing[i] == "unknown"){
    data$housing[i] = "yes"
  }
}


data$job = as.factor(data$job)
data$marital = as.factor(data$marital)
data$education = as.factor(data$education)
data$housing = as.factor(data$housing)

final = c(1,2,3,4,6,8,9,10,12,14,15,16,17,18,21) # selecting the columns to be used in the cleaned dataset
f_data = data[,final] 
View(f_data)
write.csv(f_data,"Bank-cleaned.csv")

######### EDA through GRAPHS
ggplot(data = data, aes(age))+geom_histogram(aes(fill = factor(y)))
#the distribution of the age over the target variable is quite similar, so it wont add much variance to the data

ggplot(data = data, aes(job,y, fill = y))+geom_bar(stat = "identity") # retired and student have the highest rate of saying yes
table(data$job,data$y)

ggplot(data = data, aes(education,y, fill = y))+geom_bar(stat = "identity")

ggplot(data = data, aes(month,y, fill = y))+geom_bar(stat = "identity") ### per percentage of conversion march oct sep and dec have the highest percentage

ggplot(data = data, aes(duration))+geom_histogram(aes(fill = factor(y)))

ggplot(data = data, aes(marital,y, fill = y))+geom_bar(stat = "identity")

ggplot(data = data, aes(day_of_week,y, fill = y))+geom_bar(stat = "identity") #### the distribution is almost uniform


###################### variable importance ########################
#create a task
trainTask <- makeClassifTask(data = f_data,target = "y")

im_feat <- generateFilterValuesData(trainTask, method = c("chi.squared"))
plotFilterValues(im_feat,n.show = 15)

########################### Bank should reconsider calling these customer for the campaign #########################

# having the bank priority(profit) in mind we build this function that calculates how much a category(demographic)
# makes a loss or a profit(one an average) for the bank.

# as we dont know how much profit does the bank makes per term deposit that the customer buys, so we took the assumed
# value of 10 dollars of profit on an average per sale of the product that the bank makes.

# we also assumed that bank gets a loss on average of 1 dollar per customer to sell the product due to repeated calls. 
profit_per_sale<-10
cost_per_call<-1
######
factor_var<-c("job","marital","education","default","housing","loan","contact","month","poutcome")

for(column in factor_var){
  col_data<-data[,c(column,'y')]
  col_data_table<-table(factor(col_data[,1]))
  factors_in_column<-names(col_data_table)
  for(afactor in factors_in_column){
    data_no<- subset(col_data,y=="no" & col_data[1]==afactor)
    data_yes<- subset(col_data,y=="yes" & col_data[1]==afactor)
    profit_per_category<-nrow(data_yes)*profit_per_sale-(((nrow(data_yes)+nrow(data_no))*cost_per_call))
    if(profit_per_category<0){
      cat("Bank should either not call or rethink about contacting:",column,":",afactor,":",profit_per_category,"\n")
    }
  }
}
#### the class per attributes whose average profit is less trhan 0 will be displayed as the result of above code.


#### MODEL BUILDING USING DIFFERENT ALGORITHMS AND TECHNIQUES TO PREDICT THE CUSTOMERS WHO BUYS THE PRODUCT #####

############### LOGISTIC REGRESSION BASE MODEL ################

f_data = read.csv("Bank-cleaned.csv")
f_data=f_data[,-1]
View(f_data)

##############DATA CLEANED BASE MODEL################
set.seed(89)
sample=sample.split(f_data$y,SplitRatio = 0.7)
train=subset(f_data,sample==T)
valid=subset(f_data,sample==F)

sample1=sample.split(valid$y,SplitRatio = 0.65)
validation=subset(valid,sample1==T)
test=subset(valid,sample1==F)

# Create Model
model=glm(data=train,y~.,family=binomial())
summary(model)

#Running Model on Test Set.
predct=predict(model,validation)

#Confusion Matrix
table(validation$y,predct>0.5)

#Measures
accuracy.meas(validation$y, predct)
roc.curve(validation$y, predct, plotit = F)

############## UNDERSAMPLING MODEL ##############

data_balanced_under <- ovun.sample(y ~ ., data = train, method = "under", N = 6496, seed = 89)$data
table(data_balanced_under$y)

# Create Model
model=glm(data=data_balanced_under,y~.,family=binomial())
summary(model)

#Running Model on Test Set.
predct=predict(model,validation)

#Confusion Matrix
table(validation$y,predct>0.5)

#Measures
accuracy.meas(validation$y, predct)
roc.curve(validation$y, predct, plotit = F)

Acc=(7391+612)/(7391+919+430+612)
print(Acc)

####### OVERSAMPLING MODEL #########
#Constructing OverSampled training Set
data_balanced_over <- ovun.sample(y ~ ., data = train, method = "over",N = 47512)$data
table(data_balanced_over$y)

# Create Model
model=glm(data=data_balanced_under,y~.,family=binomial())
summary(model)

#Running Model on Test Set.
predct=predict(model,validation)

#Confusion Matrix
table(validation$y,predct>0.5)

#Measures
accuracy.meas(validation$y, predct)
roc.curve(validation$y, predct, plotit = F)

Acc=(7383+654)/(7343+920+441+654)
print(Acc)

########### SMOTE MODEL ############

#Constructing Smote training Set
data_balanced_smote=SMOTE(y ~ .,train, perc.over = 200,perc.under=100,k=5)
table(data_balanced_smote$y)

# Create Model
model=glm(data=data_balanced_smote,y~.,family=binomial())
summary(model)

#Running Model on Test Set.
predct=predict(model,validation)

#Confusion Matrix
table(validation$y,predct>0.5)

#Measures
accuracy.meas(validation$y, predct)
roc.curve(validation$y, predct, plotit = F)


####################################### DECISION TREES #########################
rm(list = ls())
##############DATA CLEANED BASE MODEL################
f_data = read.csv("Bank-cleaned.csv")
f_data=f_data[,-1]
View(f_data)
set.seed(89)
sample=sample.split(f_data$y,SplitRatio = 0.7)
f_data_training=subset(f_data,sample==T)
valid=subset(f_data,sample==F)

#Constructing Validation and Test Set.
sample1=sample.split(valid$y,SplitRatio = 0.65)
f_data_validation=subset(valid,sample1==T)
f_data_testing=subset(valid,sample1==F)

dt_model<- rpart(y ~., data = f_data_training)
fancyRpartPlot(dt_model)
summary(dt_model)
predictions <- predict(dt_model, f_data_validation, type = "prob")[,2]
table(f_data_validation$y,predictions>0.5)

accuracy.meas(f_data_validation$y, predictions)
roc.curve(f_data_validation$y, predictions, plotit = F)

############################## DEALING WITH THE IMBALANCED CLASS PROBLEM #######################

## 1 UNDERSAMPLING
table(f_data$y)
data_balanced_under <- ovun.sample(y ~ ., data = f_data_training, method = "under", seed = 89,  N = 6496)
table(data_balanced_under$data$y)
dt_model<- rpart(y ~., data = data_balanced_under$data)
fancyRpartPlot(dt_model)
summary(dt_model)

predictions <- predict(dt_model, f_data_validation, type = "prob")[,2]

table(f_data_validation$y,predictions>0.5)

accuracy.meas(f_data_validation$y, predictions)
roc.curve(f_data_validation$y, predictions, plotit = F)


## 2 OVERSAMPLING
table(f_data$y)
data_balanced_over <- ovun.sample(y ~ ., data = f_data_training, method = "over", N = 51000)
table(data_balanced_over$data$y)
dt_model_o<- rpart(y ~., data = data_balanced_over$data)
fancyRpartPlot(dt_model_o)
summary(dt_model_o)

predictions <- predict(dt_model_o, f_data_validation, type = "prob")[,2]

table( f_data_validation$y,predictions>0.5)

accuracy.meas(f_data_validation$y, predictions)
roc.curve(f_data_validation$y, predictions, plotit = F)

### 3. SMOTE
data_SMOTE = SMOTE(y~.,data = f_data_training, perc.over = 200, k = 6, perc.under = 150)
table(data_SMOTE$y)
dt_model = rpart(y ~., data = data_SMOTE)
fancyRpartPlot(dt_model)
summary(dt_model)

predictions <- predict(dt_model, f_data_validation, type = "prob")[,2]

table(f_data_validation$y,predictions>0.5)

accuracy.meas(f_data_validation$y, predictions)
roc.curve(f_data_validation$y, predictions, plotit = F)


######################################## Random Forest #################################
f_data = read.csv("Bank-cleaned.csv")
f_data=f_data[,-1]
####### CLEANED DATA
set.seed(89)
sample = sample.split(f_data$y, SplitRatio = 0.7)
f_data_training = subset(f_data,sample==T)
f_data_testingVali = subset(f_data,sample==F)

# ##################### validation set
sample1 = sample.split(f_data_testingVali$y, SplitRatio = 0.6)
f_data_validation = subset(f_data_testingVali,sample1==T)
f_data_testing = subset(f_data_testingVali,sample1==F)

ranTree_c=randomForest(y~.,data=f_data_training,nTrees=500,mtry = 3)
predictions_c=predict(ranTree_c,f_data_validation,type = "prob")[,2]

table(f_data_validation$y,predictions_c>0.5)

accuracy.meas(f_data_validation$y, predictions_c)
roc.curve(f_data_validation$y, predictions_c, plotit = F)

## 1 UNDERSAMPLING
table(f_data$y)
data_balanced_under <- ovun.sample(y ~ ., data = f_data_training, method = "under", seed = 89,  N = 6496)
ranTree_u=randomForest(y~.,data=data_balanced_under$data,nTrees=500,mtry=3)
predictions_u=predict(ranTree_u,f_data_validation, type = "prob")[,2]

table(f_data_validation$y,predictions_u>0.5)

accuracy.meas(f_data_validation$y, predictions_u>0.5)
roc.curve(f_data_validation$y, predictions_u, plotit = F)

#final testing on test set
predictions_u=predict(ranTree_u,f_data_testing, type = "prob")[,2]

table(f_data_testing$y,predictions_u>0.5)

accuracy.meas(f_data_testing$y, predictions_u>0.5)
roc.curve(f_data_testing$y, predictions_u, plotit = F)


### 2 OVERSAMPLING
data_balanced_over <- ovun.sample(y ~ ., data = f_data_training, method = "over", seed = 89, N = 51000)
ranTree_o=randomForest(y~.,data=data_balanced_over$data,nTrees=500,mtry=3)
predictions_o=predict(ranTree_o,f_data_validation, type = "prob")[,2]

table(f_data_validation$y,predictions_o>0.4)

accuracy.meas(f_data_validation$y, predictions_o>0.4)
roc.curve(f_data_validation$y, predictions_o, plotit = F)

### SMOTE
data_SMOTE = SMOTE(y~.,data = f_data_training, perc.over = 200, k = 6, perc.under = 150)
table(data_SMOTE$y)
ranTree_smot=randomForest(y~.,data=data_SMOTE,nTrees=500,mtry=3)
predictions_smot=predict(ranTree_smot,f_data_validation, type = "prob")[,2]

table(f_data_validation$y,predictions_smot>0.4)
accuracy.meas(f_data_validation$y, predictions_smot>0.4)
roc.curve(f_data_validation$y, predictions_smot, plotit = F)


############################################## NAIVE BAYES ###############################################
rm(list = ls())
################### Base Model
data=read.csv('Bank-cleaned.csv')
data=data[,-1]

set.seed(89)
sample=sample.split(data$y,SplitRatio = 0.7) 
train=subset(data,sample==T)
test=subset(data,sample==F)

sample_val=sample.split(test$y,SplitRatio = 0.65)
valid_set=subset(test,sample_val==T)
test_set=subset(test,sample_val==F)

classifier1= naiveBayes(x = train[,-15],y = train[,15])

predict=predict(classifier1,valid_set[,-15])
table(predict)
table(predict,valid_set[,15],dnn=c('predicted','actual'))

x=table(predict,valid_set$y,dnn = list('predicted','actual'))
confusionMatrix(x,mode='prec_recall',positive = 'yes')
roc.curve(predict,valid_set$y,plotit = F)

### UNDERSAMPLING
under_sample=ovun.sample(y~.,data = train,method = 'under',p = 0.5)$data
table(under_sample$y)

under_train=naiveBayes(x = under_sample[,-15],y =under_sample[,15] )

#undersampling model
under_model=predict(under_train,valid_set[,-15])
table(under_model)
table(under_model,valid_set[,15],dnn = list('predicted','actual'))

x=table(under_model,valid_set$y)
confusionMatrix(x,mode = "prec_recall", positive = "yes")

#ROC CURVES
roc.curve(response = valid_set$y, predicted = under_model, plotit = F)

############# oversampling
over_sample=ovun.sample(y~.,data = train,method = 'over',p=0.5,seed = 89)$data
table(over_sample$y)

over_train=naiveBayes(x = over_sample[,-15],y = over_sample[,15])

over_model=predict(over_train,valid_set[,-15])

table(over_model)
table(over_model,valid_set[,15],dnn=list("predicted",'actual'))

library(caret)
x=table(over_model,valid_set[,15],dnn = list('predicted','actual'))
confusionMatrix(x,mode = "prec_recall", positive = "yes")
#ROC CURVE
roc.curve(response = valid_set$y,predicted = over_model, plotit = F)

######## SMOTE
smote=SMOTE(y~.,data = train)
table(smote$y)

#training
smote_model=naiveBayes(smote[,-15],smote[,15])

#testing
smote.predict=predict(smote_model,valid_set[,-15])
table(smote.predict)
table(smote.predict,valid_set$y,dnn=list('predicted','actual'))
prop.table(table(smote.predict,valid_set$y,dnn=list('predicted','actual')))

#confusion Matrix
x=table(smote.predict,valid_set$y,dnn=list('predicted','actual'))
confusionMatrix(x,mode = 'prec_recall',positive = 'yes')

#Roc
roc.curve(response = valid_set$y,predicted = smote.predict, plotit = F)

############################################## XGBOOST ###################################################
rm(list = ls())
f_data = read.csv("Bank-cleaned.csv")
f_data=f_data[,-1]
############# SMOTE
set.seed(89)

sample = sample.split(f_data$y, SplitRatio = 0.7)
f_data_training = subset(f_data,sample==T)
f_data_testingVali = subset(f_data,sample==F)
sample1 = sample.split(f_data_testingVali$y, SplitRatio = 0.65)
f_data_validation = subset(f_data_testingVali,sample1==T)
f_data_testing = subset(f_data_testingVali,sample1==F)

data_SMOTE = SMOTE(y~.,data = f_data_training, perc.over = 200, k = 5, perc.under = 150)
table(data_SMOTE$y)

setDT(data_SMOTE)
setDT(f_data_validation)

#using one hot encoding 
labels <- data_SMOTE$y
ts_label <- f_data_validation$y
new_tr <- model.matrix(~.+0,data = data_SMOTE[,-c("y"),with=F]) 
new_ts <- model.matrix(~.+0,data = f_data_validation[,-c("y"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.01, gamma=1, max_depth=4, min_child_weight=1, subsample=0.7, colsample_bytree=0.6,scale_pos_weight=7.8)
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.01, gamma=1, max_depth=4, min_child_weight=1, subsample=0.7, colsample_bytree=0.6)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 200, nfold = 6, showsd = T, stratified = T, print.every_n = 10, early_stopping_rounds= 50, maximize = T, eval_metric = "auc")

max(xgbcv$best_iteration)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 199, watchlist = list(val=dtest,train=dtrain), print.every_n = 10, early_stopping_rounds = 50, maximize = T , eval_metric = "auc")
#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

table(xgbpred, ts_label)

accuracy.meas(ts_label, xgbpred)


######## UNDERSAMPLING
set.seed(89)

sample = sample.split(f_data$y, SplitRatio = 0.7)
f_data_training = subset(f_data,sample==T)
f_data_testingVali = subset(f_data,sample==F)

# ##################### validation set
sample1 = sample.split(f_data_testingVali$y, SplitRatio = 0.65)
f_data_validation = subset(f_data_testingVali,sample1==T)
f_data_testing = subset(f_data_testingVali,sample1==F)

data_balanced_under <- ovun.sample(y ~ ., data = f_data_training, method = "under", seed = 89)
table(data_balanced_under$data$y)

xdf = data_balanced_under$data

setDT(xdf)
setDT(f_data_validation)

#using one hot encoding 
labels <- xdf$y
ts_label <- f_data_validation$y
new_tr <- model.matrix(~.+0,data = xdf[,-c("y"),with=F]) 
new_ts <- model.matrix(~.+0,data = f_data_validation[,-c("y"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.01, gamma=1, max_depth=4, min_child_weight=1, subsample=0.7, colsample_bytree=0.6)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 200, nfold = 6, showsd = T, stratified = T, print.every_n = 10, early_stopping_rounds= 50, maximize = T, eval_metric = "auc")

max(xgbcv$best_iteration)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 200, watchlist = list(val=dtest,train=dtrain), print.every_n = 10, early_stopping_rounds = 50, maximize = T , eval_metric = "auc")
#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

table( ts_label,xgbpred)
accuracy.meas(ts_label, xgbpred)

############# OVERSAMPLING
set.seed(89)
sample = sample.split(f_data$y, SplitRatio = 0.7)
f_data_training = subset(f_data,sample==T)
f_data_testingVali = subset(f_data,sample==F)

# ##################### validation set
sample1 = sample.split(f_data_testingVali$y, SplitRatio = 0.65)
f_data_validation = subset(f_data_testingVali,sample1==T)
f_data_testing = subset(f_data_testingVali,sample1==F)

data_balanced_under <- ovun.sample(y ~ ., data = f_data_training, method = "over", seed = 89)
table(data_balanced_under$data$y)

xdf = data_balanced_under$data

setDT(xdf)
setDT(f_data_validation)
setDT(f_data_testing)
#using one hot encoding 
labels <- xdf$y
ts_label <- f_data_validation$y
f_label <- f_data_testing$y
new_tr <- model.matrix(~.+0,data = xdf[,-c("y"),with=F]) 
new_ts <- model.matrix(~.+0,data = f_data_validation[,-c("y"),with=F])
new_f <- model.matrix(~.+0,data = f_data_testing[,-c("y"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
f_label <- as.numeric(f_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
ftest <- xgb.DMatrix(data = new_f, label = f_label)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.01, gamma=1, max_depth=4, min_child_weight=1, subsample=0.7, colsample_bytree=0.6)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 200, nfold = 6, showsd = T, stratified = T, print.every_n = 10, early_stopping_rounds= 50, maximize = T, eval_metric = "auc")

max(xgbcv$best_iteration)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 200, watchlist = list(val=dtest,train=dtrain), print.every_n = 10, early_stopping_rounds = 50, maximize = T , eval_metric = "auc")
#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

table( ts_label,xgbpred)
accuracy.meas(ts_label, xgbpred)

#final testing on test set
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 200, watchlist = list(val=ftest,train=dtrain), print.every_n = 10, early_stopping_rounds = 50, maximize = T , eval_metric = "auc")
xgbpred <- predict (xgb1,ftest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

table( f_label,xgbpred)
accuracy.meas(f_label, xgbpred)


#################################### SUPPORT VECTOR MACHINES ###############################
rm(list = ls())
setwd("E:/Machine Learning Aegis/ML_project")
data = read.csv("bank-additional.csv")
View(data)
str(data)
#################################### base model ################################

data$job = as.character(data$job)
data$marital = as.character(data$marital)
data$education = as.character(data$education)
data$housing = as.character(data$housing)

# for job replacing Unknownwith Admin(max frequency)
for(i in 1:length(data$job)){
  if(data$job[i] == "unknown"){
    data$job[i] = "admin."
  }
}

# for marital replacing Unknown with married(max frequency)
for(i in 1:length(data$marital)){
  if(data$marital[i] == "unknown"){
    data$marital[i] = "married"
  }
}

# for education replacing Unknown with university.degree(max frequency)
for(i in 1:length(data$education)){
  if(data$education[i] == "unknown"){
    data$education[i] = "university.degree"
  }
}

# for housing replacing Unknown with yes(max frequency)
for(i in 1:length(data$housing)){
  if(data$housing[i] == "unknown"){
    data$housing[i] = "yes"
  }
}


data$job = as.factor(data$job)
data$marital = as.factor(data$marital)
data$education = as.factor(data$education)
data$housing = as.factor(data$housing)

final = c(1,2,3,4,6,8,9,10,12,14,15,16,17,18,21)
f_data1 = data[,final]
View(f_data1)
str(f_data1)

set.seed(89)
library(caTools)
sample = sample.split(f_data1$y, SplitRatio = 0.7)
f_data1_training = subset(f_data1,sample==T)
f_data1_testingVali = subset(f_data1,sample==F)

# ##################### validation set
sample1 = sample.split(f_data1_testingVali$y, SplitRatio = 0.65)
f_data1_validation = subset(f_data1_testingVali,sample1==T)
f_data1_testing = subset(f_data1_testingVali,sample1==F)

#Using radial kernel
svm_clean_model = svm(y~., data = f_data1_training)
summary(svm_clean_model)
predict_svm_clean_model = predict(svm_clean_model, f_data1_validation)
table(predict_svm_clean_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_clean_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_clean_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = f_data1_training, kernel = "radial", ranges = list(cost=10^(-1:1),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_clean_model = svm(y~., data = f_data1_training, cost = 1, gamma =0.1 )
summary(svm_clean_model)
predict_svm_clean_model = predict(svm_clean_model, f_data1_validation)
table(predict_svm_clean_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_clean_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_clean_model, plotit = F)

#Using polynomial kernel
svm_clean_model = svm(y~., data = f_data1_training,kernel="poly")
summary(svm_clean_model)
predict_svm_clean_model = predict(svm_clean_model, f_data1_validation)
table(predict_svm_clean_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_clean_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_clean_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = f_data1_training, kernel = "poly", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_clean_model = svm(y~., data = f_data1_training,kernel="poly", cost = 1, gamma =0.1 )
summary(svm_clean_model)
predict_svm_clean_model = predict(svm_clean_model, f_data1_validation)
table(predict_svm_clean_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_clean_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_clean_model, plotit = F)

############################## DEALING WITH THE IMBALANCED CLASS PROBLEM #######################

## 1 UNDERSAMPLING
table(f_data1$y)
data_balanced_under <- ovun.sample(y ~ ., data = f_data1_training, method = "under", seed = 89)
table(data_balanced_under$data$y)

#using radial kernel
svm_undersample_model = svm(y~., data = data_balanced_under$data)
summary(svm_undersample_model)
predict_svm_undersample_model = predict(svm_undersample_model, f_data1_validation)
table(predict_svm_undersample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_undersample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_undersample_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = data_balanced_under$data, kernel = "radial", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_undersample_model = svm(y~., data = data_balanced_under$data, cost = 1, gamma = 0.1)
summary(svm_undersample_model)
predict_svm_undersample_model = predict(svm_undersample_model, f_data1_validation)
table(predict_svm_undersample_model, f_data1_validation[,"y"])
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_undersample_model, plotit = F)

#using polynomial kernel
svm_undersample_model = svm(y~., data = data_balanced_under$data, kernel="poly")
summary(svm_undersample_model)
predict_svm_undersample_model = predict(svm_undersample_model, f_data1_validation)
table(predict_svm_undersample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_undersample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_undersample_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = data_balanced_under$data, kernel = "poly", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_undersample_model = svm(y~., data = data_balanced_under$data, kernel="poly", cost = 1, gamma = 0.1)
summary(svm_undersample_model)
predict_svm_undersample_model = predict(svm_undersample_model, f_data1_validation)
table(predict_svm_undersample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_undersample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_undersample_model, plotit = F)


## 2 OVERSAMPLING
table(f_data1$y)
data_balanced_over <- ovun.sample(y ~ ., data = f_data1_training, method = "over" )
table(data_balanced_over$data$y)

#using radial kernel
svm_oversample_model = svm(y~., data = data_balanced_over$data)
summary(svm_oversample_model)
predict_svm_oversample_model = predict(svm_oversample_model, f_data1_validation)
table(predict_svm_oversample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_oversample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_oversample_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = data_balanced_over$data, kernel = "radial", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_oversample_model = svm(y~., data = data_balanced_over$data, cost= 1, gamma=0.1 )
summary(svm_oversample_model)
predict_svm_oversample_model = predict(svm_oversample_model, f_data1_validation)
table(predict_svm_oversample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_oversample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_oversample_model, plotit = F)

#using polynomial kernel
svm_oversample_model = svm(y~., data = data_balanced_over$data, kernel="poly")
summary(svm_oversample_model)
predict_svm_oversample_model = predict(svm_oversample_model, f_data1_validation)
table(predict_svm_oversample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_oversample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_oversample_model, plotit = F)
#Tunning
tuneradial = tune(svm,y~.,data = data_balanced_over$data, kernel = "poly", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_oversample_model = svm(y~., data = data_balanced_over$data, kernel="poly", cost= 1, gamma=0.1 )
summary(svm_oversample_model)
predict_svm_oversample_model = predict(svm_oversample_model, f_data1_validation)
table(predict_svm_oversample_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_oversample_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_oversample_model, plotit = F)

### 3. SMOTE
table(f_data1$y)
library(DMwR)
data_SMOTE = SMOTE(y~.,data = f_data1_training, perc.over = 200, k = 6, perc.under = 150)
table(data_SMOTE$y)
#using radial kernel
svm_smote_model = svm(y~., data = data_SMOTE)
summary(svm_smote_model)
predict_svm_smote_model = predict(svm_smote_model, f_data1_validation)
table(predict_svm_smote_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_smote_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_smote_model, plotit = F)


#Tunning
tuneradial = tune(svm,y~.,data = data_SMOTE, kernel = "radial", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#Tunned values
svm_smote_model = svm(y~., data = data_SMOTE, cost = 1, gamma = 0.1)
summary(svm_smote_model)
predict_svm_smote_model = predict(svm_smote_model, f_data1_validation)
table(predict_svm_smote_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_smote_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_smote_model, plotit = F)


# using polynomial kernel
svm_smote_model = svm(y~., data = data_SMOTE, kernel = "poly")
summary(svm_smote_model)
predict_svm_smote_model = predict(svm_smote_model, f_data1_validation)
table(predict_svm_smote_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_smote_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_smote_model, plotit = F)

#Tunning
tuneradial = tune(svm,y~.,data = data_SMOTE, kernel = "poly", ranges = list(cost=10^(-1:2),gamma = c(0.1,0.25,0.5,1,2)))
summary(tuneradial)
#tunned values
svm_smote_model = svm(y~., data = data_SMOTE, kernel = "poly", cost = 1, gamma = 0.1)
summary(svm_smote_model)
predict_svm_smote_model = predict(svm_smote_model, f_data1_validation)
table(predict_svm_smote_model, f_data1_validation[,"y"])
confusionMatrix(predict_svm_smote_model, f_data1_validation[,"y"],mode = 'prec_recall',positive = 'yes')
#Roc
roc.curve(response = f_data1_validation[,"y"],predicted = predict_svm_smote_model, plotit = F)