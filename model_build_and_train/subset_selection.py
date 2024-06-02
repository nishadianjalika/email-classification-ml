import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def select_subset():
    file_path = 'generated_csv_files/emails_converted_to_csv_output.csv'
    data = pd.read_csv(file_path)
    # print(data['from_'].value_counts().head(50))

    data = data.dropna() #37 blank rows removed
    data = data.dropna(subset=['from_'])
    # Remove all rows where 'from_' column is empty
    data = data[data['from_'].str.strip() != '']

    import numpy as np
    data.replace('N/A', np.nan, inplace=True)
    data.dropna(inplace=True)

    print(data['from_'].value_counts().head(100))

    top_senders = data['from_'].value_counts().head(100).index


    # Convert Index to DataFrame
    filtered_data = data[data['from_'].isin(top_senders)]
    # print(filtered_data.shape)
    # print(filtered_data.value_counts)

    # Save DataFrame to CSV
    filtered_data.to_csv('generated_csv_files/top_100_senders_output.csv', index=False)
    
    distinct_users = filtered_data['from_'].nunique()
    print(distinct_users)

    return filtered_data

def data_distribution(filtered_data):
    import matplotlib.pyplot as plt

    # Distribution of the number of emails per sender
    email_counts = filtered_data['from_'].value_counts()

    plt.figure(figsize=(15, 10))
    truncated_labels = [label[:5] for label in email_counts.index]  # Truncate labels to 15 characters
    print(truncated_labels)

    plt.bar(truncated_labels, email_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel('Senders')
    plt.ylabel('Number of Emails')
    plt.title('Distribution of Number of Emails per Sender')
    # plt.show()
    plt.savefig("plots/subset_distribution")

def get_avg_length():
    file_path = 'generated_csv_files/top_30_senders_output.csv'
    df = pd.read_csv(file_path)

    df['body_length'] = df['body'].apply(len)
    average_length = df['body_length'].mean()
    print(average_length)

if __name__ == "__main__":
    filtered_data = select_subset()
    data_distribution(filtered_data)
    # get_avg_length()