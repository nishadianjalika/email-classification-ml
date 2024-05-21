from pre_process_sent_items import read_and_filter_csv_for_given_folder
from extract_email_content import process_emails_for_selected_users
from extract_email_content import parse_into_emails
import pandas as pd

def main():
    # # Filter and create CSV for 'sent_items' folder (or define required folder)
    # csv_file = 'generated_csv_files/selected_folder_email_counts_per_user.csv'
    # sent_item_user_file = 'generated_csv_files/users_with_sent_items_folder.csv'
    # sent_items_df = read_and_filter_csv_for_given_folder(csv_file, 'sent_items') #filerting csv for 'sent_items' folder only
    # sent_items_df.to_csv(sent_item_user_file, index=False)
    # print(f'Filtered data saved to users_with_sent_items_folder.csv')

    #   # Define the path to the maildir folder
    # maildir_path = '../maildir/'  
    # selected_users = sent_items_df['User'].tolist()
    output_csv = 'generated_csv_files/full_sent_items_emails.csv'
    # initial_emails = process_emails_for_selected_users(maildir_path, selected_users, output_csv) #done
    
    #reading email content using code chunks taken from https://github.com/anthdm/ml-email-clustering/tree/master
    initial_emails_df = pd.read_csv(output_csv)
    initial_emails_df.dropna(inplace=True)

    email_df = pd.DataFrame(parse_into_emails(initial_emails_df.email_content))
    email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
    # print(email_df)
    
    email_converted_to_csv_output = 'generated_csv_files/emails_converted_to_csv_output.csv'
    email_df.to_csv(email_converted_to_csv_output, index=False)
    print(f'Processed emails saved to {email_converted_to_csv_output}')

    

if __name__ == "__main__":
    main()