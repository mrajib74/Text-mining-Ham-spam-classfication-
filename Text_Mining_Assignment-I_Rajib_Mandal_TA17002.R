library(quanteda)
library(caret)
library(RTextTools)
library(RColorBrewer)
library(ROCR)
library(ggplot2)

#########################################
# Read the file
#####################################
setwd("D:\\Rajib\\XLRI\\Textmining\\Assignment")
corp <- read.csv("smsspam.csv",header = TRUE,stringsAsFactors = FALSE)
str(corp)
set.seed(1234)
corp<-corp[sample(nrow(corp)),]

# Check the number of spam and ham messages
table(corp$class)
round(prop.table(table(corp$class))*100, digits = 1)
# ############################
# # Generte Barplot
# ############################
theme_set(theme_bw())
ggplot(aes(x=class),data=corp) +
  geom_bar(fill="green",width=0.5)
corphamspam <- corpus(corp$text)
docvars(corphamspam) <-corp$class 

#######################
# Get total token in the corpous
######################
corpoustoken<-corphamspam
sum(ntoken(corpoustoken,
           removeNumbers=TRUE,
           remove_punct = TRUE,
           remove_url=TRUE,
           remove_hyphens =TRUE,
           remove_symbols = TRUE,
           remove_separators = TRUE, 
           remove_numbers = TRUE))


#Add metadata to the Corpus object
metadoc(corphamspam, 'language') <- "english"


###################################
### spam world cloud
###################################
spam.plot <- corpus_subset(corphamspam, docvar1 == "spam") 


##############################
## No of tokens in Spam
token.spam.plot<-spam.plot
sum(ntoken(token.spam.plot,
           remove_punct = TRUE,
           removeNumbers=TRUE,
           remove_url=TRUE,
           remove_hyphens =TRUE,
           remove_symbols = TRUE,
           remove_separators = TRUE,
           remove_numbers = TRUE))
############################

spam.plot <- dfm(spam.plot, tolower = TRUE,
                 removeNumbers=TRUE,
                 remove_punct = TRUE, 
                 remove_url=TRUE,
                 removeNumbers=TRUE,
                 remove_hyphens =TRUE,
                 remove_symbols = TRUE,
                 remove_separators = TRUE, 
                 remove_numbers = TRUE,
                 remove=c("lt","gt","ur","ü","po","cs",letters,stopwords("english")),
                 valuetype="fixed",
                 verbose=TRUE)

topfeatures(spam.plot, 50) 
spam.col <- brewer.pal(10, "BrBG")

spam.cloud <- textplot_wordcloud(spam.plot, min_count = 16, 
                                 random_order = FALSE,random_color = FALSE,
                                 rotation = 0.1,
                                 min_size=0.5,color = spam.col)  
title("Spam Wordcloud", col.main = "grey14")
################################



###################################
### ham world cloud
###################################
ham.plot <- corpus_subset(corphamspam, docvar1 == "ham")  
##############################
# No of token in ham
##########################
token.ham.plot<-ham.plot

sum(ntoken(token.ham.plot,remove_punct = TRUE,
           removeNumbers=TRUE,
           remove_url=TRUE,
           remove_hyphens =TRUE,
           remove_symbols = TRUE,
           remove_separators = TRUE, 
           remove_numbers = TRUE))
###################################

ham.plot <- dfm(ham.plot, tolower = TRUE,
                 remove_punct = TRUE, 
                 remove_url=TRUE,
                 removeNumbers=TRUE,
                 remove_hyphens =TRUE,
                 remove_symbols = TRUE,
                 remove_separators = TRUE, 
                 remove_numbers = TRUE,
                remove=c("lt","gt","ur","ü","po","cs",letters,stopwords("english")),
               valuetype="fixed",
               verbose=TRUE)
topfeatures(ham.plot, 50) 
ham.col <- brewer.pal(10, "BrBG") 
par(mar=rep(0.4, 4))
plot(0:1, 0:1, type = "n", axes = FALSE, ann = FALSE)
ham.cloud <- textplot_wordcloud(ham.plot, min_count =50,min_size=0.9,max_size = 4,
                                random_order = FALSE,random_color = FALSE,
                                rotation = 0.1,
                                fixed_aspect = TRUE,
                                color = ham.col) 


title(main = list("ham Wordcloud", cex = 1.5,col = "grey14", font = 1))
################################

dfmhamspam <- dfm(corphamspam, 
                  tolower = TRUE,
                  stem = TRUE,
                  removeNumbers=TRUE,
                  remove_punct = TRUE,
                  remove_url=TRUE,
                  remove_hyphens =TRUE,
                  remove_symbols = TRUE,
                  remove_separators = TRUE,
                  remove_twitter = TRUE,
                  valuetype="fixed",
                  verbose=TRUE
                  )

dfmhamspam <- dfm_trim(dfmhamspam, min_docfreq = 5, max_docfreq = 500,sparsity = NULL) 
summary(dfmhamspam)
docvars(corphamspam) <-corp$class

intTraingSet<-as.integer(nrow(corp)*0.80)
intTestSet<-intTraingSet+1
corp.train<-corp[1:intTraingSet,]
corp.test<-corp[intTestSet:nrow(corp),]
dfmhamspam.train<- dfmhamspam[1:intTraingSet]


dfmhamspam.test<-dfmhamspam[intTestSet:nrow(corp),]
nb <- textmodel_nb(dfmhamspam.train,corp.train[,1],smooth = 1 )

pred_data <- predict(nb, dfmhamspam.test)
predicted_class <- pred_data$nb.predicted 
tb<-table(predicted_class,corp.test$class)
tb
accuracy<-sum(diag(tb))/sum(tb)
accuracy


predvec <- ifelse(predicted_class=="ham", 1, 0)
realvec <- ifelse(corp.test$class=="ham", 1, 0)
pred <- prediction(predvec,realvec)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = "green", lwd = 3,
     xlab="False Positive Rate",ylab="True Positive Rate")
abline(a = 0, b = 1, lwd = 2, lty = 2)
perf.auc <- performance(pred, measure = "auc")
unlist(perf.auc@y.values)

#######################################################
##Predict new dataset
#######################################################

ab<-read.csv("smsspam_predict.csv",header=TRUE,stringsAsFactors=FALSE)
test_dfm <-dfm(corpus(ab$text),tolower = TRUE,
               remove_punct = TRUE, 
               remove_url=TRUE,
               remove_hyphens =TRUE,
               remove_symbols = TRUE,
               remove_separators = TRUE, 
               remove_numbers = TRUE,
               remove=c("lt","gt","ur","ü","po","cs",letters,stopwords("english")),
               valuetype="fixed",
               verbose=TRUE)


test_dfm <- dfm_select(test_dfm, dfmhamspam.train,selection = "keep")

## Make predictions on test set
predvec_new_1<- predict(nb, test_dfm, type = "response")
predicted_class <- predvec_new_1$nb.predicted 
predvec<-ifelse(predicted_class=="ham",1,0)
ab$class<-predicted_class
write.csv(ab,"spsspam_new_predict.csv")

