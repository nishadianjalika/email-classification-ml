import os
import pandas as pd

def load_full_email_content(email_path):
    try:
        with open(email_path, 'r', encoding='utf-8', errors='replace') as file:
            email_content = file.read()
    except Exception as e:
        print(f"Error reading file {email_path}: {e}")
        return ""
    
    return email_content

# def process_emails_for_selected_users(maildir_path, selected_users, output_csv):
#     email_data = []

#     for user in selected_users:
#         user_path = os.path.join(maildir_path, user, 'sent_items')
#         if os.path.isdir(user_path):
#             for email_file in os.listdir(user_path):
#                 email_path = os.path.join(user_path, email_file)
#                 if os.path.isfile(email_path):
#                     print(f"loading email content for {user} {email_file}")
#                     email_content = load_full_email_content(email_path)
#                     email_data.append([user, email_file, email_content])

#     # Save the extracted data to a CSV file
#     df = pd.DataFrame(email_data, columns=['user', 'email_file', 'email_content'])
#     df.to_csv(output_csv, index=False)
#     print(f'Processed emails saved to {output_csv}')
#     return df

def process_emails_for_selected_users(maildir_path, selected_users, output_csv, chunk_size=100):
    email_data = []
    
    for user in selected_users:
        user_path = os.path.join(maildir_path, user, 'sent_items')
        if os.path.isdir(user_path):
            email_files = os.listdir(user_path)
            for i, email_file in enumerate(email_files):
                email_path = os.path.join(user_path, email_file)
                if os.path.isfile(email_path):
                    print(f"Loading email content for {user} {email_file}")
                    email_content = load_full_email_content(email_path)
                    email_data.append([user, email_file, email_content])
                
                # Save data in chunks
                if len(email_data) >= chunk_size:
                    save_to_csv(email_data, output_csv)
                    email_data = []  # Reset the list to free memory

    # Save any remaining data
    if email_data:
        save_to_csv(email_data, output_csv)

    print(f'Processed emails saved to {output_csv}')

def save_to_csv(data, output_csv):
    df = pd.DataFrame(data, columns=['user', 'email_file', 'email_content'])
    if not os.path.isfile(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, mode='a', header=False, index=False)

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

