from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#1. Data representation using TF-IDF
#TF-IDF score represents the relative importance of a term in the document and the entire corpus. 
#TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), 
#the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents 
#in the corpus divided by the number of documents where the specific term appears.

def convert_to_tf_idf_word_level(trainDF, train_x, valid_x, test_x):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    xtest_tfidf = tfidf_vect.transform(test_x)
    return xtrain_tfidf, xvalid_tfidf, xtest_tfidf


def convert_to_tf_idf_ngram_level(trainDF, train_x, valid_x, test_x):
    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram, xtest_tfidf_ngram

def convert_to_tf_idf_characters_level(trainDF, train_x, valid_x, test_x):
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
    xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x) 
    return xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars, xtest_tfidf_ngram_chars