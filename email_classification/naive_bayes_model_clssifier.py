from model_train import train_model
def train_nb_model():

    train_x, train_y, test_x, test_y = generate_train_test_splits()
    xtrain_count = convert_to_count_vec(train_x)
    xvalid_count = test_x

    xtrain_tfidf = 

    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    print ("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    print ("NB, WordLevel TF-IDF: ", accuracy)