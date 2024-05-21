from sklearn import metrics
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

#Call this method by passing relevant clssifier and train dataset (transferred to tf-idf/word_embed/etc)
def train_model(classifier, feature_vector_train, train_y, feature_vector_valid, val_y, is_neural_net=False, apply_weights = True ):
    #to address the class imbalance issue
    #try with and without class_weights and compare the results

    class_weights = compute_class_weight(class_weight="balanced", 
                         classes=np.unique(train_y), 
                         y=train_y
                         )
    
    # fit the training dataset on the classifier
    if apply_weights:
        classifier.fit(feature_vector_train, 
                   train_y,
                   class_weight = class_weights)
    else:
        classifier.fit(feature_vector_train, 
                   train_y)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, val_y)
 