library(neuralnet)

wine_data <- read.csv("C:/Users/Rajat Singh Panwar/Desktop/Wine Classification/wine.data.txt", header=F)
fix(wine_data)
maxs <- apply(wine_data[,2:14], 2, max)
mins <- apply(wine_data[,2:14], 2, min)
scaled_data <- as.data.frame(scale(wine_data[,2:14], center = mins, scale = maxs - mins))
library(dummies)
p <- dummy.data.frame(wine_data[1], names=c("V1"), sep="_")
#p <- class.ind(wine_data$V1)
scaled_data <- cbind(p, scaled_data)
names(scaled_data)[1] <- "O1"
names(scaled_data)[2] <- "O2"
names(scaled_data)[3] <- "O3"
fix(scaled_data)

#Splitting dataset into Train and Test Dataset

#train <- subset(scaled_data[1:150,])
#test <- subset(scaled_data[151:178,])

smp_size <- floor(0.75 * nrow(scaled_data))

set.seed(123)
train_ind <- sample(seq_len(nrow(scaled_data)), size=smp_size)
print(train_ind)

train <- scaled_data[train_ind,]
test <- scaled_data[-train_ind,]

#Creating formula for neuralnet

feats <- names(scaled_data[,4:16])

# Concatenate strings 
f <- paste(feats,collapse=' + ')
f <- paste('O1+O2+O3~',f)

# Convert to formula
f <- as.formula(f)
f

#Our neuralnet model

nn <- neuralnet(f, train, hidden=c(5), linear.output=FALSE)


# Compute Predictions off Test Set
predicted.nn.values <- compute(nn,test[,4:16])

# Check out net.result
x <- predicted.nn.values$net.result
print(head(x))

#Choosing the class having maximum probability
predictions <- as.data.frame(apply(x,1,FUN=which.max))
print(head(predictions))
fix(predictions)

plot(nn)

#Computing Accuracy
real <- as.data.frame(wine_data[-train_ind, 1])

tf <- predictions == real
accFreq <- table(tf)
print(accFreq)
trueVals = accFreq["TRUE"]
falseVals = accFreq["FALSE"]
accuracy <- trueVals/(trueVals+falseVals)

print(accuracy)