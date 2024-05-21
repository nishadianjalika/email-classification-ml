from emails_per_user_analysis import count_emails_in_folders
from total_email_count_per_user import count_total_emails_per_user
from selected_folder_wise_email_count import count_emails_in_selected_folders
import os

def main():
    # count number of emails for each user for thier sub folders
    maildir_path = '../maildir/'
    email_counts_per_user_folder_csv_file = 'generated_csv_files/email_counts_per_user_folder.csv'
    count_emails_in_folders(maildir_path, email_counts_per_user_folder_csv_file)

    # Get the total email counts for each user
    csv_file = 'generated_csv_files/total_email_counts_per_user.csv'
    total_email_plot = 'plots/total_email_counts_per_user.png'
    count_total_emails_per_user(maildir_path, csv_file, total_email_plot)

    # Get the email counts for each user in the specified folders
    folders = ['inbox', '_sent_email', 'sent', 'sent_items']
    csv_file = 'generated_csv_files/selected_folder_email_counts_per_user.csv'
    output_dir = './plots'
    os.makedirs(output_dir, exist_ok=True)
    email_counts = count_emails_in_selected_folders(maildir_path, folders, csv_file, output_dir)

if __name__ == "__main__":
    main()