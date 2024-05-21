from model_train import train_model
from generate_train_test_data import generate_train_test_splits
from feat_rep_using_count_vect import convert_to_count_vec

def train_rf_model(train_df):
    train_x, train_y, test_x, test_y , val_x, val_y= generate_train_test_splits()
    #create required feature representations (count vector/ tf-idf/etc) and pass it tio train_model() method

    # Naive Bayes on Count Vectors
    accuracy = train_model() #fill parameters required
    print ("SVM, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model()#fill parameters required
    print ("SVM, WordLevel TF-IDF: ", accuracy)