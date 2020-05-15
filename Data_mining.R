#Importing Libraries
library(MASS)
library(mlr)
library(randomForest)
library(tm)
library(e1071)
libs <- c("tm","slam","dplyr","tidyr","purrr","readr","plyr","class","caret","wordcloud","rpart","tree","mlr")
libs <- c("tm","plyr","class","readr","purrr")
lapply(libs, require, character.only = TRUE)
library(caret)
library(wordcloud)
library(reshape)
library(rpart)
library(tree)
library(dplyr)
library(tidyr)
options(stringsAsFactors = FALSE)

#Paths for NewsGroups
Newsgroup       <-                  c("comp.sys.ibm.pc.hardware", "sci.electronics", "talk.politics.guns", "talk.politics.misc")
News_Pathname   <-                  "/Users/prajwalg/Desktop/DATA Mining/Newsgroups"
Training_News_folder   <-            "/Users/prajwalg/Desktop/DATA Mining/Newsgroups"

Func_folder_mining     <-      function(subfolder) {
tibble(file =dir(subfolder, full.names = TRUE)) %>%
mutate(full_text =map(file, read_file)) %>%        
transmute(id = basename(file), full_text) %>%   #replacing existing columns with new columns
unnest(full_text)
}


#subfolder
raw_text_corpus     <-          tibble(folder = dir(Training_News_folder, full.names = TRUE)) %>%
mutate(output_folder   =     map(folder, Func_folder_mining)) %>%
unnest(cols = c(output_folder)) %>%
transmute(full_text   =       full_text,Newsgroup_Class = basename(folder))



#Creating Corpus for training and testing
News_data_hardware_frequency     <-             subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "comp.sys.ibm.pc.hardware")
News_data_electronic_frequency   <-             subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "sci.electronics")
News_data_politics_frequency     <-             subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "talk.politics.guns")
News_data_misc_frequency         <-             subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "talk.politics.misc")

#Set the seed 
set.seed (30)

#Division of raw data
for  (k in 1:100){
  sample_hardware_frequency    <-      sample.int(n=100, size=70)
  sample_electronic_frequency  <-      sample.int(n=100, size=70)
  sample_politics_frequency    <-      sample.int(n=100, size=70)
  sample_misc_frequency        <-      sample.int(n=100, size=70)}

News_training.set <- rbind(News_data_hardware_frequency[sample_hardware_frequency,], News_data_electronic_frequency[sample_electronic_frequency,], News_data_politics_frequency[sample_politics_frequency,], News_data_misc_frequency[sample_misc_frequency,])
News_testing.set  <- rbind(News_data_hardware_frequency[-sample_hardware_frequency,], News_data_electronic_frequency[-sample_electronic_frequency,], News_data_politics_frequency[-sample_politics_frequency,], News_data_misc_frequency[-sample_misc_frequency,])

actual_newsdata_test_class    <-                         News_testing.set$Newsgroup_Class
News_testing.set_text         <-                         News_testing.set$full_text

actual_newsdata_train_class      <-                     News_training.set$Newsgroup_Class
News_training.set_text           <-                     News_training.set$full_text



#Corpus with the training subset
Func_build_Corpus <- function(Newsgroup_Class, train.subset){
tp = toString(Newsgroup_Class)
train.new <- subset(train.subset,train.subset$Newsgroup_Class == tp)
sourse.corps <- Corpus(VectorSource(train.new$full_text))
term.data.mat <- TermDocumentMatrix(sourse.corps)
result <- list(name=Newsgroup_Class, tdm = term.data.mat)
}



#Method to build corpus with the training subset
Func_bindCandidateToTDM <- function(data_matrix){
  source_matrix <- t(data.matrix(data_matrix[["tdm"]])) 
  source_dataframe <- as.data.frame(source_matrix,StringsAsFactors = FALSE)
  source_dataframe <- cbind(source_dataframe, rep(data_matrix[["name"]], nrow(source_dataframe)))
  colnames(source_dataframe)[ncol(source_dataframe)] <- "Newsgroup_Class"
  return(source_dataframe)
}
# Training the News Data frames
News_data_training.tdm <- lapply(Newsgroup, Func_build_Corpus, News_training.set)
News_data_training_cand_TDM <- lapply(News_data_training.tdm, Func_bindCandidateToTDM)
News_data_training.stack <- do.call(rbind.fill,News_data_training_cand_TDM)
News_data_training.stack[is.na(News_data_training.stack)] <- 0
col_idx <- grep("Newsgroup_Class", names(News_data_training.stack))
News_data_training.stack <- News_data_training.stack[, c((1:ncol(News_data_training.stack))[-col_idx],col_idx)]

# Testing the News Data frames
News_data_testing.tdm <- lapply(Newsgroup, Func_build_Corpus, News_testing.set)
News_data_testing_cand_TDM <- lapply(News_data_testing.tdm, Func_bindCandidateToTDM)
News_data_test.stack <- do.call(rbind.fill,News_data_testing_cand_TDM)
News_data_test.stack[is.na(News_data_test.stack)] <- 0
col_idx <- grep("Newsgroup_Class", names(News_data_test.stack))
News_data_test.stack <- News_data_test.stack[, c((1:ncol(News_data_test.stack))[-col_idx],col_idx)]


#Calculate the overall TextnDataMatrix from the data
Overall.TextnDataMatrix <- lapply(Newsgroup, Func_build_Corpus, raw_text_corpus)
Overall_cand_TextnDataMatrix <- lapply(Overall.TextnDataMatrix, Func_bindCandidateToTDM)
Overall.TextnDataMatrix <- do.call(rbind.fill,Overall_cand_TextnDataMatrix)
Overall.TextnDataMatrix[is.na(Overall.TextnDataMatrix)] <- 0
col_idx <- grep("Newsgroup_Class", names(Overall.TextnDataMatrix))
Overall.TextnDataMatrix <- Overall.TextnDataMatrix[, c((1:ncol(Overall.TextnDataMatrix))[-col_idx],col_idx)]


Overall.TextnDataMatrix.dataframe=Overall.TextnDataMatrix
#Removing the last document type column from the matrix
Overall.TextnDataMatrix.dataframe$Newsgroup_Class=NULL

#8. Sorting the TextnDataMatrix.stack.dataframe in decreasing order using sum of all columns 
Top_200_words_in_dataframe=head((sort(colSums(Overall.TextnDataMatrix.dataframe), decreasing = TRUE)),200)
print(Top_200_words_in_dataframe)

#9. Filtering the data with word length between 4 and 20

Total_no_of_colnmns = as.numeric(ncol(Overall.TextnDataMatrix))
Total_word_occurence = rep(0, each=Total_no_of_colnmns-1)
for(i in 1:(Total_no_of_colnmns-1)){
  Total_word_occurence[i] = sum(as.vector(Overall.TextnDataMatrix[i]))   
}
word_count_data = cbind.data.frame(names(Overall.TextnDataMatrix.dataframe), as.numeric(Total_word_occurence))
ordered_word_count_data = word_count_data[order(word_count_data[,2], decreasing = TRUE),]
names(ordered_word_count_data) = c("Words","Total Occurences")

ordered_word_count_data$Words_Length = nchar(as.character(ordered_word_count_data[,1]))
count = 1
for(i in 1:length(ordered_word_count_data[,1])){
  if(ordered_word_count_data[i,3] >= 4 && ordered_word_count_data[i,3] <= 20){
    print(ordered_word_count_data[i,])
    count = count + 1
    if(count == 201){ 
      break
    }
  }
}

write.csv(ordered_word_count_data,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/ordered_word_count_data.csv')





#10. Dividing the data set(matrix) to train and test in 70:30 ratio.
TrainingDataSet <- sample(nrow(Overall.TextnDataMatrix), ceiling(nrow(Overall.TextnDataMatrix) * 0.7))
TestingDataSet <- sample(1:nrow(Overall.TextnDataMatrix)) [-TrainingDataSet]




#11. Applying Knn Modelling 
Overall.TextnDataMatrix.newsType <- Overall.TextnDataMatrix[,"Newsgroup_Class"]
TrainingDataSet.nl <- Overall.TextnDataMatrix[, !colnames(Overall.TextnDataMatrix) %in% "Newsgroup_Class"]
x.train.knn=TrainingDataSet.nl[TrainingDataSet,]
x.test.knn=TrainingDataSet.nl[TestingDataSet,]
y.train.knn=Overall.TextnDataMatrix.newsType[TrainingDataSet]
y.test.knn=Overall.TextnDataMatrix.newsType[TestingDataSet]
set.seed(6405)
knn.model.b<- knn(x.train.knn,x.test.knn,y.train.knn)
conf.mat.b <- table("Predictions" = knn.model.b, Actual =y.test.knn )
conf.mat.b
(acuracy <- sum(diag(conf.mat.b)) / length(TestingDataSet) * 100) 
calc_precision <- conf.mat.b[1,1]/sum(conf.mat.b[,1]) * 100
calc_precision
calc_recall    <- conf.mat.b[1,1]/sum(conf.mat.b[1,]) * 100
calc_recall





#11. Random Forest
class(Overall.TextnDataMatrix)
train <- Overall.TextnDataMatrix[TrainingDataSet, ]
train.tp <- t(train)
write.csv(train.tp,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/train.tp.csv')
test <- Overall.TextnDataMatrix[TestingDataSet, ]
train$Newsgroup_Class <- as.factor(train$Newsgroup_Class)
train.tp <- t(train)
write.csv(train.tp,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/train.tp1.csv')
test$Newsgroup_Class=as.factor(test$Newsgroup_Class)
set.seed(7000)
classifier.rf<- randomForest(x = train,y = train$Newsgroup_Class,nTree = 500)
rf.pred <- predict(classifier.rf,test)
conf.mat.rf <- table(test[,"Newsgroup_Class"], rf.pred)
acuracy.rf <- sum(diag(conf.mat.rf)) / length(TestingDataSet) * 100
acuracy.rf
calc_precision <- conf.mat.rf[1,1]/sum(conf.mat.rf[,1]) * 100
calc_precision
calc_recall    <- conf.mat.rf[1,1]/sum(conf.mat.rf[1,]) * 100
calc_recall








#Naive Bayes

#Creating Dataset

total_hardware_word_count = 0
total_electronics_word_count = 0
total_guns_word_count = 0
total_misc_word_count = 0
total_vocabulary_count = 0

Func_NaiveByaes_train_data(News_data_training.stack)

#News_data_testing.set_text
#NB_pred <- unlist(lapply(News_training.set,Func_Naive_Bayes))
#NB_pred_Conf_Mat <-table(actual_newsdata_test_class,NB_pred)

N.Bayes_pred <- unlist(lapply(News_testing.set_text,Func_Naive_Bayes))
N.Bayes_pred_Confusion_Mat <-table(actual_newsdata_test_class, N.Bayes_pred)

#computing accuracy
N.Bayes_accuracy <- sum(diag(N.Bayes_pred_Confusion_Mat))/sum(N.Bayes_pred_Confusion_Mat)*100
N.Bayes_accuracy
Naive_Bayes_Precision <- N.Bayes_pred_Confusion_Mat[1,1]/sum(N.Bayes_pred_Confusion_Mat[,1]) * 100
Naive_Bayes_Recall    <- N.Bayes_pred_Confusion_Mat[1,1]/sum(N.Bayes_pred_Confusion_Mat[1,]) * 100
Naive_Bayes_Precision
Naive_Bayes_Recall




#Method to train the data set
Func_NaiveByaes_train_data <- function(data_matrix) {
  
  
  # Calculating the Hardware news data row sums
  computed_hardware_frequency <- subset(data_matrix,data_matrix$Newsgroup_Class == "comp.sys.ibm.pc.hardware")
  computed_hardware_frequency <- as.data.frame(computed_hardware_frequency)
  #writing into the file
  write.csv(computed_hardware_frequency,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/computed_hardware_frequency.csv')
  col_idx <- grep("Newsgroup_Class", names(computed_hardware_frequency))
  hardware_frequency_sums <- computed_hardware_frequency[, c((1:ncol(computed_hardware_frequency))[-col_idx])]
  demo = data.frame(hardware_frequency_sums[0,])
  hardware_frequency_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(hardware_frequency_sums, na.rm = T))
  #Filtering the data
  head(hardware_frequency_sums)
  #Removing zero values
  hardware_frequency_sums <- as.data.frame(hardware_frequency_sums)
  hardware_row_sums_No_Zeros <- hardware_frequency_sums[hardware_frequency_sums$frequency != 0, ]
  #sum of all the frequncy in hardware news folder
  total_hardware_word_count <- sum(hardware_frequency_sums$frequency)
  #Filtering data
  head(hardware_row_sums_No_Zeros)
  write.csv(hardware_row_sums_No_Zeros,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/hardware_frequency_sums_ZeroEx.csv')
  
  
  # Calculating the Electronics news data row sums
  computed_electronics_frequency <- subset(data_matrix,data_matrix$Newsgroup_Class == "sci.electronics")
  computed_electronics_frequency <- as.data.frame(computed_electronics_frequency)
  write.csv(computed_electronics_frequency,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/computed_electronics_frequency.csv')
  col_idx <- grep("Newsgroup_Class", names(computed_electronics_frequency))
  electronics_frequncy_sums <- computed_electronics_frequency[, c((1:ncol(computed_electronics_frequency))[-col_idx])]
  demo = data.frame(electronics_frequncy_sums[0,])
  electronics_frequncy_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(electronics_frequncy_sums, na.rm = T))
  #Removing zero values
  electronics_frequncy_sums <- as.data.frame(electronics_frequncy_sums)
  electronics_row_sums_No_Zeros <- electronics_frequncy_sums[electronics_frequncy_sums$frequency != 0, ]
  #sum of all the frequncy in electronics news folder
  total_electronics_word_count <- sum(electronics_frequncy_sums$frequency)
  head(electronics_frequncy_sums)
  #writing into the file
  write.csv(electronics_frequncy_sums,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/electronics_row_sums_Zero.csv')
  
  
  
  # Calculating the Politics and guns news data row sums
  computed_pol_guns_frequency <- subset(data_matrix,data_matrix$Newsgroup_Class == "talk.politics.guns")
  col_idx <- grep("Newsgroup_Class", names(computed_pol_guns_frequency))
  guns_frequncy_sums <- computed_pol_guns_frequency[, c((1:ncol(computed_pol_guns_frequency))[-col_idx])]
  demo = data.frame(guns_frequncy_sums[0,])
  guns_frequncy_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(guns_frequncy_sums, na.rm = T))
  #Removing zero values
  guns_frequncy_sums <- as.data.frame(guns_frequncy_sums)
  guns_row_sums_No_Zeros <- guns_frequncy_sums[guns_frequncy_sums$frequency != 0, ]
  #sum of all the frequncy in pol and guns news folder
  total_guns_word <- sum(guns_frequncy_sums$frequency)
  head(guns_frequncy_sums)
  #writing into the file
  write.csv(guns_frequncy_sums,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/guns_row_sums_Zero.csv')
  
  
  
  # Calculating the Misc news data row sums
  computed_misc_frequency <- subset(data_matrix,data_matrix$Newsgroup_Class == "talk.politics.misc")
  col_idx <- grep("Newsgroup_Class", names(computed_misc_frequency))
  misc_frequncy_sums <- computed_misc_frequency[, c((1:ncol(computed_misc_frequency))[-col_idx])]
  demo = data.frame(misc_frequncy_sums[0,])
  misc_frequncy_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(misc_frequncy_sums, na.rm = T))
  #Removing zero values
  misc_frequncy_sums <- as.data.frame(misc_frequncy_sums)
  misc_row_sums_No_Zeros <- misc_frequncy_sums[misc_frequncy_sums$frequency != 0, ]
  #sum of all the frequncy in misc news folder
  total_misc_word_count <- sum(misc_frequncy_sums$frequency)
  head(misc_frequncy_sums)
  write.csv(misc_frequncy_sums,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/misc_row_sums_Zero.csv')
  
  
  # Getting the totol word count of the whole data matrix
  total_row_freq_count <- data_matrix[, c((1:ncol(data_matrix))[-col_idx])]
  total_row_freq_count <- slam::col_sums(total_row_freq_count, na.rm = T)
  total_row_freq_count <- as.data.frame(total_row_freq_count)
  total_vocabulary_count <- nrow(total_row_freq_count)
  #writing into the file
  write.csv(total_row_freq_count,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/total_row_sums.csv')
  
  
  #Hardware news probability calculations using the NB formula
  #Creating data frame of the matrix
  hardware_row_sums_No_Zeros <- as.data.frame(hardware_row_sums_No_Zeros)
  #Calculating the probablity- rowsum+1/total words + total word count
  hardware_news_probs <- (hardware_row_sums_No_Zeros$frequency+1)/(total_hardware_word_count+total_vocabulary_count) 
  #Removing all the zeros and bind
  hardware_row_prob_No_Zeros <- cbind(hardware_row_sums_No_Zeros,hardware_news_probs)
  write.csv(hardware_row_prob_No_Zeros,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/hardware_probs.csv')
  #train the data
  hardware_traindata <- read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/hardware_probs.csv",header=T,sep=",")
  hardware_traindata[c(2:4)]
  write.csv(hardware_traindata[c(2:4)],'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/trained_hardware_probs.csv')
  
  #Electronics news probability calculations using the NB formula
  #Creating data frame of the matrix
  electronics_row_sums_No_Zeros <- as.data.frame(electronics_row_sums_No_Zeros)
  #Calculating the probablity- rowsum+1/total words + total word count
  electronics_news_probs <- (electronics_row_sums_No_Zeros$frequency+1)/(total_electronics_word_count+total_vocabulary_count) 
  electronics_row_prob_No_Zeros <- cbind(electronics_row_sums_No_Zeros,electronics_news_probs)
  write.csv(electronics_row_prob_No_Zeros,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/electronics_news_probs.csv')
  elec_traindata <- read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/electronics_news_probs.csv",header=T,sep=",")
  elec_traindata[c(2:4)]
  write.csv(elec_traindata[c(2:4)],'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/trained_electronics_probs.csv')
  
  
  #Politics and Guns news probability calculations using the NB formula
  #Creating data frame of the matrix
  guns_row_sums_No_Zeros <- as.data.frame(guns_row_sums_No_Zeros)
  #Calculating the probablity- rowsum+1/total words + total word count
  guns_news_probs <- (guns_row_sums_No_Zeros$frequency+1)/(total_guns_word+total_vocabulary_count) 
  guns_row_prob_No_Zeros <- cbind(guns_row_sums_No_Zeros,guns_news_probs)
  write.csv(guns_row_prob_No_Zeros,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/guns_news_probs.csv')
  guns_traindata <- read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/guns_news_probs.csv",header=T,sep=",")
  guns_traindata[c(2:4)]
  write.csv(guns_traindata[c(2:4)],'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/trained_guns_news_probs.csv')
  
  
  #Misc Probability Calculations
  misc_row_sums_No_Zeros <- as.data.frame(misc_row_sums_No_Zeros)
  #Calculating the probablity- rowsum+1/total words + total word count
  misc_news_probs <- (misc_row_sums_No_Zeros$frequency+1)/(total_misc_word_count+total_vocabulary_count) 
  misc_row_prob_ZeroEx <- cbind(misc_row_sums_No_Zeros,misc_news_probs)
  write.csv(misc_row_prob_ZeroEx,'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/misc_news_probs.csv')
  misc_traindata <- read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/misc_news_probs.csv",header=T,sep=",")
  write.csv(misc_traindata[c(2:4)],'/Users/prajwalg/Desktop/DATA Mining/Newsgroups/trained_misc_probs.csv')
  
}

Func_Naive_Bayes <- function(string) {
  list = unlist(strsplit(string," "))
  hardware_probability = 0
  electronics_probability = 0
  guns_probability = 0
  misc_probability= 0
  
  hardware_word_probability <- format(read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/hardware_probs.csv",header=T,sep=","), scientific = FALSE)
  electronics_word_probability <- format(read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/electronics_news_probs.csv",header=T,sep=","), scientific = FALSE)
  guns_word_probability <- format(read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/guns_news_probs.csv",header=T,sep=","), scientific = FALSE)
  misc_word_probability <-  format(read.csv("/Users/prajwalg/Desktop/DATA Mining/Newsgroups/misc_news_probs.csv",header=T,sep=","), scientific = FALSE)
  
  
  #splitting the list into words
  list = as.list(strsplit(string, '\\s+')[[1]])
  
  
  for (word in list) {
    if(length(hardware_word_probability[hardware_word_probability$colname== word,]$hardware_probability) != 0 && word %in% hardware_word_probability$colname){
      hardware_probability = hardware_probability + log(as.numeric(hardware_word_probability[hardware_word_probability$colname== word,]$hardware_probability))
    }else{
      hardware_probability = hardware_probability + log((1/(total_hardware_word_count+total_vocabulary_count)))
    }
    if(length(electronics_word_probability[electronics_word_probability$colname== word,]$electronics_probability) != 0 && word %in% electronics_word_probability$colname){
      electronics_probability = electronics_probability + log(as.numeric(electronics_word_probability[electronics_word_probability$colname== word,]$electronics_probability))
    }else{
      electronics_probability = electronics_probability + log((1/(total_electronics_word_count+total_vocabulary_count))) 
    }
    if(length(guns_word_probability[guns_word_probability$colname== word,]$guns_probability) != 0 && word %in% guns_word_probability$colname){
      guns_probability = guns_probability + log(as.numeric(guns_word_probability[guns_word_probability$colname== word,]$guns_probability))
    }else{
      guns_probability = guns_probability + log((1/(total_guns_word_count+total_vocabulary_count)))
    } 
    if(length(misc_word_probability[misc_word_probability$colname== word,]$misc_probability) != 0 && word %in% misc_word_probability$colname){
      misc_probability = misc_probability + log(as.numeric(misc_word_probability[misc_word_probability$colname== word,]$misc_probability))
    }else{
      misc_probability = misc_probability + log((1/(total_misc_word_count+total_vocabulary_count)))
    }
  }
  #summation of all probabilities
  results <- c(hardware_probability,electronics_probability,guns_probability,misc_probability)
  #taking the maximum of the calculated probabilities
  Maximum <- max(results)
  #Calculating the largest and predicting the newsGroup
  if (Maximum== hardware_probability){
    group_classification <- "comp.sys.ibm.pc.hardware"
  } else if(Maximum== electronics_probability) {
    group_classification <- "sci.electronics"
  } else if (Maximum== guns_probability) {
    group_classification <- "talk.politics.guns"
  } else {
    group_classification <- "talk.politics.misc"
  } 
  
  return(group_classification)
}


#####################ROBUST EVALUATION###################################



#Clean Corpus of Term Data Matrix
cleanCorpusTDM <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp,removeNumbers)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  return(corpus.tmp)
}

#Build TDM for Robust Evaluation
buildRobustTDM <- function(newsGroupClassRobust, path){
  source.dir <- sprintf("%s/%s",path,newsGroupClassRobust)
  source.cor <- Corpus(DirSource(directory = source.dir))
  source.cor.cl <- cleanCorpusTDM(source.cor)
  source.tdm <- TermDocumentMatrix(source.cor.cl)
  result_list <- list(name=newsGroupClassRobust, tdm = source.tdm)
}

robustTDM <- lapply(Newsgroup, buildRobustTDM, path = News_Pathname)

#Bind newsGroup to Robust TDM
bindNewsgroupToRobustTDM <- function(robustTDM){
  source.mat <- t(data.matrix(robustTDM[["tdm"]]))
  source.df <- as.data.frame(source.mat,StringsAsFactors = FALSE)
  
  source.df <- cbind(source.df, rep(robustTDM[["name"]], nrow(source.df)))
  colnames(source.df)[ncol(source.df)] <- "newsGroupClassRobust"
  return(source.df)
}

newsRobustTDM <- lapply(robustTDM, bindNewsgroupToRobustTDM)


robustTDM.stack.R <- do.call(rbind.fill,newsRobustTDM)
robustTDM.stack.R[is.na(robustTDM.stack.R)] <- 0
colIndx <- grep("newsGroupClassRobust", names(robustTDM.stack.R))
robustTDM.stack.R <- robustTDM.stack.R[, c((1:ncol(robustTDM.stack.R))[-colIndx],colIndx)]
dim(robustTDM.stack.R)

robustTDM.stack.R.df=robustTDM.stack.R
robustTDM.stack.R.df$newsGroupClassRobust=NULL

# Creation of Training and Test Set Indexes
robustTrain.indx.R <- sample(nrow(robustTDM.stack.R), ceiling(nrow(robustTDM.stack.R) * 0.7))
robustTest.indx.R  <- sample(1:nrow(robustTDM.stack.R)) [-robustTrain.indx.R]

#KNN Model using Robust Evaluation
robustTDM.newsGroupClass <- robustTDM.stack.R[,"newsGroupClassRobust"]
robustTDM.stack.nl <- robustTDM.stack.R[, !colnames(robustTDM.stack.R) %in% "newsGroupClassRobust"]
set.seed(20)
x.robustTrain.knn.R=robustTDM.stack.nl[robustTrain.indx.R,]
x.robustTest.knn.R=robustTDM.stack.nl[robustTest.indx.R,]
y.robustTrain.knn.R=robustTDM.newsGroupClass[robustTrain.indx.R]
y.robustTest.knn.R=robustTDM.newsGroupClass[robustTest.indx.R]
robustKNN.model.R<- knn(x.robustTrain.knn.R,x.robustTest.knn.R,y.robustTrain.knn.R)
conf.mat.R <- table("Predictions" = robustKNN.model.R, Actual =y.robustTest.knn.R )
#acuracy
(acuracy.knn  <- sum(diag(conf.mat.R)) / length(robustTest.indx.R) * 100)
knn_precision <- conf.mat.R[1,1]/sum(conf.mat.R[,1]) * 100
knn_recall    <- conf.mat.R[1,1]/sum(conf.mat.R[1,]) * 100
knn_precision
knn_recall

#Random Forest using Robust Evaluation
trainRobust <- robustTDM.stack.R[robustTrain.indx.R, ]
testRobust  <- robustTDM.stack.R[robustTest.indx.R, ]
trainRobust$newsGroupClassRobust <- as.factor(trainRobust$newsGroupClassRobust)
testRobust$newsGroupClassRobust=as.factor(testRobust$newsGroupClassRobust)
set.seed(50)
classifier.rf.ro <- randomForest(x = trainRobust,y = trainRobust$newsGroupClassRobust,nTree = 5000)
rf.ro.pred <- predict(classifier.rf.ro,testRobust)
cm.rf.ro <- table(testRobust[,"newsGroupClassRobust"], rf.ro.pred)
acuracy.rf.ro <- (sum(diag(cm.rf.ro)) / length(robustTest.indx.R)) * 100
acuracy.rf.ro
rand_precision <- cm.rf.ro[1,1]/sum(cm.rf.ro[,1]) * 100
rand_recall    <- cm.rf.ro[1,1]/sum(cm.rf.ro[1,]) * 100
rand_precision
rand_recall


#Naive Bayes




