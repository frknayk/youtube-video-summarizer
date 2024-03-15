import os

def save_transcript_to_file(transcript_text, file_name):
    temp_folder = 'temp'
    file_path = os.path.join(temp_folder, file_name)

    # Create the 'temp' directory if it doesn't exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(transcript_text)

        return file_path
    except Exception as e:
        return str(e)
