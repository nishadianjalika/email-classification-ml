import os
import csv
import matplotlib.pyplot as plt

def count_total_emails_per_user(maildir_path, csv_file, plot_file):
    # List to store the results
    total_email_counts = []

    # Traverse the maildir directory
    for user in os.listdir(maildir_path):
        user_path = os.path.join(maildir_path, user)
        if os.path.isdir(user_path):
            total_emails = 0
            for subfolder in os.listdir(user_path):
                subfolder_path = os.path.join(user_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Count the number of email files in the subfolder
                    email_count = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
                    # Add to the total email count for the user
                    total_emails += email_count
            # Append the result to the list
            total_email_counts.append([user, total_emails])

    save_to_csv(total_email_counts, csv_file)
    plot_email_counts(total_email_counts, plot_file)
    return total_email_counts

def save_to_csv(email_counts, csv_file):
    # Write the results to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Total_No_of_Emails'])
        writer.writerows(email_counts)

def plot_email_counts(email_counts, plot_file):
    # Extract user names and their corresponding email counts
    users = [item[0] for item in email_counts]
    counts = [item[1] for item in email_counts]

    # Plotting the bar chart
    plt.figure(figsize=(14, 8))  # Adjust size for better readability
    plt.bar(users, counts, color='skyblue')
    plt.ylabel('Total Number of Emails')
    plt.xlabel('Users')
    plt.title('Total Email Count per User')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(plot_file)  # Save the plot
    # plt.show()  # Optionally display the plot


