import os
import csv

def count_emails_in_folders(maildir_path, csv_file):
    # List to store the results
    email_counts = []

    # Traverse the maildir directory
    for user in os.listdir(maildir_path):
        user_path = os.path.join(maildir_path, user)
        if os.path.isdir(user_path):
            for subfolder in os.listdir(user_path):
                subfolder_path = os.path.join(user_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Count the number of email files in the subfolder
                    email_count = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
                    # Append the result to the list
                    email_counts.append([user, subfolder, email_count])

    save_to_csv(email_counts, csv_file)

    return email_counts

def save_to_csv(email_counts, csv_file):
    # Write the results to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Subfolder', 'No_of_Emails'])
        writer.writerows(email_counts)


