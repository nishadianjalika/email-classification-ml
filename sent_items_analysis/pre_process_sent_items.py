import pandas as pd
import matplotlib.pyplot as plt

def read_and_filter_csv_for_given_folder(csv_file, folder_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Filter out the rows for the specified folder
    filtered_df = df[df['Folder'] == folder_name]
    total_users = filtered_df.shape[0]
    users_without_sent_items = filtered_df[filtered_df['Total_No_of_Emails'] == 0].shape[0]
    percentage_without_sent_items = (users_without_sent_items / total_users) * 100
    
    print(f"Total users: {total_users}")
    print(f"Users without 'sent_items' folder: {users_without_sent_items}")
    print(f"Percentage without 'sent_items' folder: {percentage_without_sent_items:.2f}%")
    
    filtered_df = filtered_df[filtered_df['Total_No_of_Emails'] > 0]
    # ToDo: Remove other users based on defined threshold count for emails

    # Analyze and plot the 'sent_items' folder data
    plot_preprocessed_sent_items(filtered_df)

    return filtered_df

def plot_preprocessed_sent_items(filtered_df):
    
    # Remove users without 'sent_items' folder
    filtered_df = filtered_df[filtered_df['Total_No_of_Emails'] > 0]
    
    # Plotting the bar chart
    users = filtered_df['User']
    counts = filtered_df['Total_No_of_Emails']

    plt.figure(figsize=(14, 8))  # Adjust size for better readability
    plt.bar(users, counts, color='skyblue')
    plt.ylabel('Total Number of Emails')
    plt.xlabel('Users')
    plt.title('Total Email Count per User in sent_items folder after pre-processing')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()
    plot_file = 'plots/total_email_count_per_user_in_sent_items_after_preprocess.png'
    plt.savefig(plot_file)  # Save the plot
    # plt.show()  # Optionally display the plot
    print(f'Plot saved to {plot_file}')


