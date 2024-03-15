from youtube_api import fetch_transcript, save_transcript_to_file
from summarizer import summarize_video_transcript

def summary_from_url(video_url:str, summary_length=-1):
        # Fetch the transcript
        transcript = fetch_transcript(video_url)
        if not transcript:
            print("Transcript not found")
        # Save the transcript to a file (optional)
        file_name = "video_transcript.txt"
        file_path = save_transcript_to_file(transcript, file_name)
        # Summarize the video transcript
        summary = summarize_video_transcript(transcript, summary_length)
        print(summary)
        return summary, file_path

video_url = "https://youtu.be/13CZPWmke6A"
# video_url = "https://youtu.be/pd0JmT6rYcI"
summary_from_url(video_url)