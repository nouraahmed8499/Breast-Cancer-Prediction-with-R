#importing and exploring the data set

data <- read.csv("breast cancer.csv",header=FALSE, sep=",")

str(data)
head(data)
summary(data)

#data pre processing 


names(data) <- c('Id','Cl_thickness','Cell_size','Cell_shape','Marg_adhesion','Epith_c_size','Bare_nuclei','Bl_cromat','Normal_nucleoli','Mitoses','Class')
data$Class[data$Class == 2] = 0
data$Class[data$Class == 4] = 1
# replacing NA with 0 and then replacing 0 values with the mean (all of them in Bate_nuclei column)
summary(data)
sum(data == '?')
# Bate_nuclei range is 1-10 so it's safe to fill missing data with 0 and then replace 0 with mean
data[data == '?'] <- 0
data$Bare_nuclei <- as.numeric(data$Bare_nuclei)
data$Bare_nuclei[data$Bare_nuclei == 0 ] <- mean(data$Bare_nuclei, na.rm =TRUE)
#The Id column is filtered out as it is not needed for designing the classifier
data$Id <- NULL
str(data)
summary(data)

#splitting the data set into train and test sets

library(caTools)
set.seed(123)    
split=sample.split(data, SplitRatio = 0.8)
training_set=subset(data,split==TRUE)
test_set=subset(data,split==FALSE)
dim(training_set) 

#feature scaling 


training_set[,1:9] = scale(training_set[,1:9])
test_set[,1:9] = scale(test_set[,1:9])

#logistic regression 

#building the classifier 
Classifier = glm(formula = Class ~ ., 
                 family = binomial, 
                 data = training_set)
#predicting the test set results
prob_pred = predict(Classifier, type = 'response', newdata = test_set[1:9])
prob_pred
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred
#confusion Matrix
cm = table(test_set[,10], y_pred)
cm
#calculating the accuracy without applying k-fold 
accuracy = (cm[1,1] + cm[2,2]/(cm[1,1] + cm[2,2]+ cm[1,2] + cm[2,1]))
accuracy
#applying k_fold Cross Validation
library(caret)
folds = createFolds(training_set$Class, k = 10)
CrossValidation = lapply(folds, function(x){
  training_fold = training_set[-x,] # taking all the training set but without the fold
  test_fold = training_set[x,]
  Classifier = glm(formula = Class ~ ., 
                   family = binomial, 
                   data = training_fold)
  prob_pred = predict(Classifier, type = 'response', newdata = test_fold[1:9])
  y_pred = ifelse(prob_pred > 0.5, 1, 0)
  cm = table(test_fold[,10], y_pred)
  accuracy = ((cm[1,1] + cm[2,2])/(cm[1,1] + cm[2,2]+ cm[1,2] + cm[2,1]))
  return(accuracy)
})
CrossValidation
accuracies = mean(as.numeric(CrossValidation))
accuracies
#Grid search (Finding the perfect hyper parameters)
library(caret)
classifier = train(form = as.factor(Class) ~ ., data = training_set, method='glm')
classifier
classifier$bestTune

#Visualization

summary(Classifier)
summ = summary(Classifier)
p_values = summ$coefficients[,4]
p_values =  p_values[p_values>0.5]
imp_cols = names(p_values)
imp_cols
set = data
plot(set)
# 2- Cell_size
plot(Class ~ Cell_size, data = set, 
     col = ifelse(set[, 10] == 1, 'green4', 'red3'), pch = 20, ylim = c(-0.2, 1),
     main = "Logistic Regression for Classification")
curve(predict(Classifier, data.frame(Cell_size = x), type = "response"), 
      add = TRUE, lwd = 3, col = "dodgerblue")
abline(v = -coef(Classifier)[1] / coef(Classifier)[2], lwd = 2)

# 5- Epith_c_size
plot(Class ~ Epith_c_size, data = set, 
     col = ifelse(set[, 10] == 1, 'green4', 'red3'), pch = 20, ylim = c(-0.2, 1),
     main = "Logistic Regression for Classification")
curve(predict(Classifier, data.frame(Epith_c_size = x), type = "response"), 
      add = TRUE, lwd = 3, col = "dodgerblue")
abline(v = -coef(Classifier)[1] / coef(Classifier)[2], lwd = 2)


