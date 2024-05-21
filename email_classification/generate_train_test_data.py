import pandas as pd
from sklearn import model_selection, preprocessing

def generate_train_test_splits():
    email_df = pd.read_csv('generated_csv_files/email_converted_to_csv_output.csv')

    #split to train and test with split of 7:3
    # but this is not good as it shuffle both train and test
    # train_x, valid_x, train_y, valid_y = model_selection.train_test_split(email_df['body'], email_df['from_'], 
    #                                                                       train_size = 0.7,
    #                                                                       test_size = 0.3,
    #                                                                       shuffle = True)


    #so manually creatinf train and test set with 7:3 
    # Determine the split index
    train_split_index = int(len(email_df) * 0.7) #70% train
    test_split_index = int(len(email_df) * 0.85)#remianing 15%-15% test and val

    # Split the data manually
    train_df = email_df.iloc[:train_split_index].reset_index(drop=True) #0 until train_split_index
    test_df = email_df.iloc[train_split_index:test_split_index].reset_index(drop=True) #from train_split_index to test_split_index
    validation_df = email_df.iloc[test_split_index:].reset_index(drop=True) # from test_split_index to end


    train_x = train_df['body']
    train_y = train_df['from_']
    test_x = test_df['body']
    test_y = test_df['from_']
    validation_x = validation_df['body']
    validation_y = validation_df['from_']

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    #encode the target column so that it can be used in machine learning models. 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    validation_y = encoder.fit_transform(validation_y)
    #check data as a sanity check
    # print(train_x[0:5])

    return train_x, train_y, test_x, test_y, validation_x, validation_y
