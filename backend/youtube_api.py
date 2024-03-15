from youtube_transcript_api import YouTubeTranscriptApi
from utils import save_transcript_to_file


def fetch_transcript(video_url):
    try:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[1]

        # Fetch the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Process the transcript data
        text_transcript = "\n".join([entry['text'] for entry in transcript])

        return text_transcript

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Define the YouTube video URL (replace with your desired video URL)
    video_url = "https://www.youtube.com/watch?v=rGgGOccMEiY&list=WL&index=2&ab_channel=CSERCambridge"
    transcript_text = fetch_transcript(video_url)
    if transcript_text:
        print("Transcript:\n")
        print(transcript_text)
        save_transcript_to_file(transcript_text, "video_transcript.txt")
    else:
        print("Failed to fetch the transcript.")
