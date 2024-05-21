from model_train import train_model
from generate_train_test_data import generate_train_test_splits
from feat_rep_using_count_vect import convert_to_count_vec
from sklearn import naive_bayes
 

def train_nb_model(train_df):

    train_x, train_y, test_x, test_y , val_x, val_y= generate_train_test_splits()

    #create required feature representations (count vector/ tf-idf/etc) and pass it tio train_model() method

    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB()) #fill parameters required
    print ("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB())#fill parameters required
    print ("NB, WordLevel TF-IDF: ", accuracy)