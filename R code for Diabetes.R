library(corrplot)
library(caret)
library(readr)
library(randomForest)
library(caret)
library(tree)
library(e1071)

rm(list = ls())


diabetes <- read.csv("C:/Users/Raja Amlan/Desktop/Data Science Projects Kaggle/Diabetes/diabetes.csv")



head(diabetes)
str(diabetes)
View(diabetes)


## We can use this to have a look at missing values
sum(is.na(diabetes$Pregnancies))
sum(is.na(diabetes$Glucose))
sum(is.na(diabetes$SkinThickness))
sum(is.na(diabetes$BMI))
count(is.na(diabetes$Outcome))

##Using the sapply function,
sapply(diabetes, function(x) sum(is.na(x)))


####Scatterplot of all the variables, this explains how the variables are correlated to each other.
pairs(diabetes, panel = panel.smooth, line.main = 3, col = c('blue'))


### Correlation plot of the variables, to understand the relation between variables or features
corrplot(cor(diabetes[, -9]), type = "lower", method = "number")

corrplot(cor(diabetes), method = "circle")


cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
# matrix of the p-value of the correlation
p.mat <- cor.mtest(diabetes)
head(p.mat[, 1:6])
options(scipen = 99)

## tl.col  and tl.srt  are used to change text colors and rotations.
corrplot(cor(diabetes), type="upper", order="hclust", tl.col="black", tl.srt=45)


# Preparing the DataSet, dividing the data in train and test sets here
set.seed(123)
n <- nrow(diabetes)
train <- sample(n, trunc(0.70*n))
diabetes_training <- diabetes[train, ]
diabetes_testing <- diabetes[-train, ]

# Training The Model: Logistic Regression
model_glm <- glm(Outcome~.,data=diabetes_training,family = binomial)

### Use step wise variable selection method to predict most important variables
step_model <- step(model_glm)

summary(model_glm)
### Some of the variables are not statistically significant hence now we will remvoe them from the model in this case
model_glm2 <- update(model_glm, ~. - SkinThickness - Insulin - Age )
summary(model_glm2)

##Plotting the model
par(mfrow = c(2,2))
plot(glm_fm2)

pred_glm<- predict(glm_fm2, data= diabetes_training, type = "response")
pred_glm

(table(ActualValue = diabetes_training$Outcome, PredictedValue = pred_glm>0.3))

pred_glm_test <- predict(glm_fm2, newdata = diabetes_testing, type = "response")
pred_glm_test

#classification matrix --
(table(ActualValue = diabetes_testing$Outcome, PredictedValue = pred_glm_test>0.5))

accuracy <- table(pred_glm_test>0.5, diabetes_testing[,"Outcome"])
accuracy_logistic<-sum(diag(accuracy))/sum(accuracy)
accuracy_logistic



######################## Now the Decision tree #####################
# Preparing the DataSet:
diabetes$Outcome <- as.factor(diabetes$Outcome)



set.seed(1000)
intrain <- createDataPartition(y = diabetes$Diabetes, p = 0.7, list = FALSE)
train <- diabetes[intrain, ]
test <- diabetes[-intrain, ]

# Training The Model
treemod <- tree(Outcome ~ ., data = train)

summary(treemod)

treemod # get a detailed text output.
plot(treemod)
text(treemod, pretty = 0)


# Testing the Model
tree_pred <- predict(treemod, newdata = test, type = "class" )
confusionMatrix(tree_pred, test$Outcome)


acc_treemod <- confusionMatrix(tree_pred, test$Outcome)$overall['Accuracy']
acc_treemod

### Applying the ensemble method: Random Forests
# Training The Model
set.seed(123)

diabetes$Outcome <- as.factor(diabetes$Outcome)

set.seed(123)
n <- nrow(diabetes)
train <- sample(n, trunc(0.70*n))
train <- diabetes[train, ]
test <- diabetes[-train, ]


diabetes_pima <- randomForest(Outcome ~., data = diabetes_training, mtry = 8, ntree=200, importance = TRUE)
diabetes_pima
# Tune only mtry parameter using tuneRF()
set.seed(1234567)      
res <- tuneRF(x = subset(train, select = -Outcome),
              y = train$Outcome,
              ntreeTry = 500,
              plot = TRUE, tunecontrol = tune.control(cross = 5))



# Random Forest Performance, accuracy is close to 92%
par(mfrow = c(1,2))

rf.predict.in <- predict(diabetes_pima, train)
rf.pred.in <- predict(diabetes_pima, train,type="prob")
confusionMatrix(train$Outcome, rf.predict.in)

diabetes_pima<- randomForest(Outcome~., data = train, importance=TRUE)
rf.predict.out <- predict(diabetes_pima, test)
rf.pred.out <- predict(diabetes_pima, test,type="prob")
confusionMatrix(test$Outcome, rf.predict.out)
rf_accuracy<-confusionMatrix(test$Outcome, rf.predict.out)$overall['Accuracy']
rf_accuracy

roc.plot(test$Outcome== "1", rf.pred.out[,2], ylab = "True Positive Rate", xlab = "False Positive Rate")$roc.vol

## Most Important Variable

importance(diabetes_pima) ### Best variable is Glucose reducing Gini by maximum

par(mfrow = c(1, 2))
varImpPlot(diabetes_pima, type = 2, main = "Variable Importance",col = 'black')
plot(diabetes_pima, main = "Error vs no. of trees grown")


################ SVM Model ###################



#Preparing the DataSet:
set.seed(1000)
intrain <- createDataPartition(y = diabetes$Outcome, p = 0.7, list = FALSE)
train <- diabetes[intrain, ]
test <- diabetes[-intrain, ]

svm_model <- tune.svm(Outcome ~., data = train, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(svm_model) # to show the results

## Best Parameters are cost 10 and gamma value to be 0.01

svm_model  <- svm(Outcome ~., data = train, kernel = "radial", gamma = 0.01, cost = 10) 
summary(svm_model)

svm_pred <- predict(svm_model, newdata = test)
confusionMatrix(svm_pred, test$Diabetes)

svm.predict.in <- predict(svm_model, train)
svm.pred.in <- predict(svm_model, train,type="prob")
confusionMatrix(train$Outcome, svm.predict.in)

svm.predict.out <- predict(svm_model, test)
svm.pred.out <- predict(svm_model, test,type="prob")
svm_accuracy<-confusionMatrix(test$Outcome, svm.predict.out)$overall['Accuracy']
svm_accuracy


############# Model Accuracy Comparison ###################

accuracy <- data.frame(Model=c("Logistic Regression","Decision Tree","Random Forest", "Support Vector Machine"), Accuracy=c(accuracy_logistic, acc_treemod, rf_accuracy, svm_accuracy ))
ggplot(accuracy,aes(x=Model,y=Accuracy)) + geom_bar(stat='identity') + theme_bw() + ggtitle('Comparison of Model Accuracy')

###So we have our decision RandomForests was the best performing algorithm here.