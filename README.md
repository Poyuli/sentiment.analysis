# sentiment.analysis

Sentiment analysis of IMDB movie reviews using word2vec and scikit-learn

We extracted the raw texts from IMDB movie reviews, and classified them to be positive if their ratings are higher than or equal to 7, negative if lower than or equal to 4. Neutral reviews are not included in this dataset. 

Both labels are balanced. There are 25000 training reviews and 25000 test reviews. Additional 50000 unlabeled training reviews are used to build the word2vec model.

We aim to build a binary classifier to predict the sentiment given raw texts of movie reviews.


# Performance Report   

Default setting: 

1. We remove stop words if unigram is used, and don't remove them if bigram or bigram/unigram is used.
2. Dict size for non-word2vec vectorizers: unigram 5k, bigram 10k, bigram/unigram 10k.
3. We does not take logarithms to tf (i.e., sublinear_tf = False).
4. For word2vec, to obtain the feature vector for each review, first we learn the vector representation of words, and then average all vectors of the words in each review.
5. For feature scaling, "standard" methods scale data to be centered around 0 with unit variance; "unsigned" methods scale data to have the max value 1 and min value 0; "signed" methods scale data to have the max value 1 and min value -1.


Following are 10-fold cross validation scores for different machine learning algorithms
(or OOB Score for Random Forest):

# Random Forest
0.82752 (n_estimators=100)

RF does not suit well in text data, since RF makes the split based on randomly selected feature. 
Text data are sparse vectors, and we may select and split based on irrelevant features.


# Gaussian Naive Bayes 
0.7348 (int counts)     
0.8008 (tf-idf)     
0.7194 (word2vec/avg(vec))

tf-idf may make distribution more Gaussian


# Multinomial Naive Bayes
0.8483 (int counts)                  
0.8528 (int counts, bigram)                
0.8488 (int counts, bigram/unigram)                 
0.8534 (binary counts)                 
0.8546 (binary counts, bigram)                 
0.8636 (binary counts, bigram/unigram)

bigram improves accuracy as it does not remove negation words.
bigram/unigram further improves accuracy as it may make feature more focused.


# Bernoulli Naive Bayes
0.8517                  
0.8496 (bigram)             
0.8606 (bigram/unigram)

Note that Bernoulli NB is different from binary multinomial NB, since Bernoulli NB penalizes absence of words by multiplying 1-p, 
where p = Prob(x_i|y) and x_i is the i-th feature which does not appear in the given sample.


# Linear SVM
0.8851 (binary counts, bigram/unigram)              
0.8873 (int counts, bigram/unigram)             
0.8802 (word2vec/avg(vec), dim=500, scaling=standard)       
0.8922 (tf-idf, bigram/unigram)     
0.8843 (tf-idf, bigram/unigram, dim=500)        
0.8845 (tf-idf, bigram/unigram, dim=500, sublinear_tf=on)       
0.8871 (tf-idf, bigram/unigram, dim=500, sublinear_tf=on, scaling=standard)             
0.8868 (tf-idf, bigram/unigram, dim=500, sublinear_tf=on, scaling=unsigned)     
0.8863 (tf-idf, bigram/unigram, dim=500, sublinear_tf=on, scaling=signed)       
0.8618 (tf-idf, bigram/unigram, dim=100, sublinear_tf=on, scaling=unsigned)

avg(word2vec) does not make sense as it also ignores the order of words.

PCA reduces the dimension based on covariance matrix, which is expensive computationally. 
LDA is also computationaly expensive when both num_feature and num_sample are large.
Besides, P(x|y) is not so Gaussian, which violates the assumption for LDA.

Truncated SVD reduces the dimension directly based on the data.

Therefore we choose truncated SVD for dimension reduction.

# Summary
Linear SVM with tf-idf and bigram/unigram vectorizing yields the best result with 89.22% accuracy

# Future work
Use doc2vec to learn sentence vectors instead of averaging word2vec vectors
