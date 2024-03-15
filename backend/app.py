from fastapi import FastAPI, HTTPException
from typing import Optional
from youtube_api import fetch_transcript, save_transcript_to_file
from summarizer import summarize_video_transcript

app = FastAPI()

@app.post("/fetch-transcript")
async def fetch_transcript_url(video_url: str):
    transcript = fetch_transcript(video_url)
    if transcript:
        return {"transcript": transcript}
    else:
        return {"error": "Failed to fetch the transcript."}

@app.post("/summarize-video")
async def summarize_video(transcript: str, summary_length: Optional[int] = 5):
    summary = summarize_video_transcript(transcript, summary_length)
    return {"summary": summary}

@app.post("/summarize-video-url")
async def summarize_video_url(video_url: str, summary_length: int = 5):
    try:
        summary, file_path = summary_from_url(video_url, summary_length)
        return {"summary": summary, "transcript_file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def summary_from_url(video_url:str, summary_length=-1):
        # Fetch the transcript
        transcript = fetch_transcript(video_url)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        # Save the transcript to a file (optional)
        file_name = "video_transcript.txt"
        file_path = save_transcript_to_file(transcript, file_name)
        # Summarize the video transcript
        summary = summarize_video_transcript(transcript, summary_length)
        return summary, file_path

if __name__ == '__main__':
    video_url = "https://www.youtube.com/watch?v=rGgGOccMEiY&list=WL&index=2&ab_channel=CSERCambridge"
    summarize_video_url(video_url)