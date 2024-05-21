from sklearn.feature_extraction.text import CountVectorizer

def convert_to_count_vec(trainDF, train_x, valid_x, test_x):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['body'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    xtest_count =  count_vect.transform(test_x)

    return xtrain_count, xvalid_count, xtest_count