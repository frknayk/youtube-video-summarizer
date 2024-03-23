# Podcast Summarizer

Podcast Summarizer is a Streamlit application that summarizes podcast episodes into concise text snippets, providing users with quick insights into the content of each episode.

## Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description

This project aims to provide a convenient way for users to get summaries of podcast episodes, helping them decide which episodes to listen to based on their interests.

The project consists of a summarization module (`summarizer.py`) containing the logic for summarizing podcast episodes, and a Streamlit application (`app.py`) that provides a user-friendly interface for interacting with the summarization functionality.

## Dependencies

The project relies on the following Python libraries:

- streamlit
- transformers
- torch
- youtube_transcript_api

You can install these dependencies using the provided `requirements.txt` file.

## Usage

To run the Podcast Summarizer application, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/podcast-summarizer.git
   cd podcast-summarizer
   ```

    2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```bash
    python -m streamlit run app.py
    ```

4. Access the application in your web browser at http://localhost:8501.


## Contributing
Contributions to the Podcast Summarizer project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
