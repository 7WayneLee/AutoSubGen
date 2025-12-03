# AutoSubGen

**AutoSubGen** is an automated movie subtitle generator designed to create English and Traditional Chinese (TC) subtitles. Whether you need independent subtitles for each language or bilingual subtitles for learning and entertainment, AutoSubGen streamlines the process for you.

## Features

-   **Automatic Transcription**: Generates English subtitles from video audio sources.
-   **Translation Support**: Automatically translates English subtitles into Traditional Chinese.
-   **Bilingual Output**: Supports generating standalone English, standalone Traditional Chinese, or bilingual (dual-language) subtitle files.
-   **Easy Integration**: Simple configuration and execution pipeline.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* **Python+**: Ensure you have Python installed.
* **FFmpeg**: Required for audio extraction and processing.
    * *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.
    * *Mac*: `brew install ffmpeg`
    * *Linux*: `sudo apt install ffmpeg`

### Installation

1.  **Install Dependencies**
    Install the required Python packages.
    ```bash
    pip install openai google-generativeai anthropic pysubs2 openai-whisper tqdm psutil
    ```

### Configuration

Before running the program, you need to configure the settings in `config.py`.

1.  Open `config.py` in your text editor.
2.  Edit the config

Once configured, you can run the generator using the main script:

```bash
python main.py
```

