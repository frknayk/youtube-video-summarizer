import streamlit as st
# Transcript
from youtube_transcript_api import YouTubeTranscriptApi
import os
# Summarization
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import re


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

def clean_transcript(transcript):
    # Remove non-speech elements (e.g., laughter, background noises)
    transcript = re.sub(r'\[.*?\]', '', transcript)

    # Correct spelling and grammar (you can use libraries like NLTK or spaCy for this)
    # Example:
    # import nltk
    # transcript = ' '.join(nltk.word_tokenize(transcript))

    # Normalize punctuation and formatting
    transcript = transcript.replace('\n', ' ')  # Remove line breaks
    transcript = re.sub(r'\s+', ' ', transcript)  # Remove extra whitespaces

    # Remove timestamps and annotations
    transcript = re.sub(r'\[\d+:\d+:\d+\]', '', transcript)

    # Handle speaker identification (if present)
    # Example: transcript = re.sub(r'Speaker\d+:', '', transcript)

    # Remove filler words and phrases
    filler_words = ['like', 'you know', 'sort of']  # Add more as needed
    for word in filler_words:
        transcript = transcript.replace(word, '')
    
    # Replace common contractions with their expanded forms
    transcript = transcript.replace("won't", "will not")
    transcript = transcript.replace("can't", "cannot")
    transcript = transcript.replace("n't", " not")
    transcript = transcript.replace("'ll", " will")
    transcript = transcript.replace("'ve", " have")
    transcript = transcript.replace("'re", " are")
    transcript = transcript.replace("'d", " would")
    transcript = transcript.replace("'s", " is")

    return transcript.strip()  # Trim leading/trailing whitespaces

def extract_video_id(url):
    """Extracts the YouTube video ID from the URL."""
    match = re.search(r"(?<=v=)[\w-]+", url)
    if match:
        return match.group(0)
    else:
        return None


def summarize_transcript(text, llama_pipeline):
    def summarize_text(llama_pipeline, system_prompt, text):
        # Format the input text with special tokens for the model
        text = f"""
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        {text}[/INST]
        """
        # Generate sequences using the pipeline with specified parameters
        sequences = llama_pipeline(text)
        # Extract the generated text from the sequences
        generated_text = sequences[0]["generated_text"]
        # Trim the generated text to remove the instruction part
        generated_text = generated_text[generated_text.find('[/INST]')+len('[/INST]'):]
        # Return the processed generated text
        return generated_text
    # Define the maximum input length for each iteration of summarization
    input_len = 1000
    # Start an infinite loop to repeatedly summarize the text
    while True:
        # Print the current length of the text
        print(len(text))
        # Call the chat function to summarize the text. Only the first 'input_len' characters are considered for summarization
        summary = summarize_text(llama_pipeline, "", "Summarize the following: " + text[0:input_len])
        if len(summary) < input_len:
            return summary
        # Concatenate the current summary with the remaining part of the text for the next iteration
        text = summary + " " + text[input_len:]

# Load the model and tokenizer
@st.cache_resource()
def load_model():
    # Define the model name to be used for the chat function
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline_llama2 = pipeline(
        "text-generation", #task
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=max_token_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return pipeline_llama2

def main():
    st.title("YouTube Video Preview")

    with st.spinner('Loading checkpoint shards of LLAMA-2'):
        pipeline_llama2 = load_model()
    st.success('Done!')

    # Input field for the YouTube video link
    youtube_url = st.text_input("Paste YouTube Video Link:")
    
    # Extract video ID from the URL
    video_id = extract_video_id(youtube_url)

    # Display video preview if video ID is found
    if video_id:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        st.video(video_url, format='video/mp4')
        video_transcript = clean_transcript(fetch_transcript(video_url))
        if video_transcript:
            # Display transcript and summary side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Transcript:")
                st.text_area(" ", video_transcript, height=400)

            with col2:
                st.subheader("Summary:")
                video_summary = summarize_transcript(video_transcript, pipeline_llama2)
                st.text_area(" ", video_summary, height=400)
                print(f"Summary:{video_summary}")   
        else:
            st.error("Failed to fetch video transcript. Please check the video ID or try again later.")

    elif youtube_url:
        st.warning("Invalid YouTube Video Link")

if __name__ == "__main__":
    main()
