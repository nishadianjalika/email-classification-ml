import os
import csv
import matplotlib.pyplot as plt

def count_emails_in_selected_folders(maildir_path, folders, csv_file, output_dir):
    # Dictionary to store the results for each folder
    email_counts = {folder: [] for folder in folders}

    # Traverse the maildir directory
    for user in os.listdir(maildir_path):
        user_path = os.path.join(maildir_path, user)
        if os.path.isdir(user_path):
            for folder in folders:
                subfolder_path = os.path.join(user_path, folder)
                email_count = 0
                if os.path.isdir(subfolder_path):
                    # Count the number of email files in the subfolder
                    email_count = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
                # Append the result to the list for the specific folder
                email_counts[folder].append([user, email_count])

    save_to_csv(email_counts, csv_file)
    plot_email_counts(email_counts, output_dir)
    return email_counts

def save_to_csv(email_counts, csv_file):
    # Write the results to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Folder', 'Total_No_of_Emails'])
        for folder, counts in email_counts.items():
            for count in counts:
                writer.writerow([count[0], folder, count[1]])

def plot_email_counts(email_counts, output_dir):
    for folder, counts in email_counts.items():
        users = [item[0] for item in counts]
        counts = [item[1] for item in counts]

        # Plotting the bar chart
        plt.figure(figsize=(14, 8))  # Adjust size for better readability
        plt.bar(users, counts, color='skyblue')
        plt.ylabel('Total Number of Emails')
        plt.xlabel('Users')
        plt.title(f'Total Email Count per User in {folder} folder')
        plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'total_email_counts_per_user_in_{folder}.png')
        plt.savefig(plot_file)  # Save the plot
        # plt.show()  # Optionally display the plot
        print(f'Plot saved to {plot_file}')

