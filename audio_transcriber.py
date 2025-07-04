from langflow.custom import Component
from langflow.io import FileInput, DropdownInput, Output, SecretStrInput
from langflow.schema import Data
import io
from openai import OpenAI


class AudioToWhisperText(Component):
    display_name = "Audio to Whisper Text"
    description = "Transcribes a WAV or OGG (Opus) file using OpenAI's Whisper model with language selection."
    icon = "mic"
    name = "AudioToWhisperText"

    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB limit for Whisper API

    inputs = [
        FileInput(
            name="audio_file",
            display_name="Audio File",
            info="Upload a WAV or OGG (Opus) audio file (max 25MB).",
            required=True,
            file_types=["wav", "ogg"]
        ),
        DropdownInput(
            name="language",
            display_name="Language",
            info="Select the language of the audio.",
            options=[
                "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
                "ar", "hi", "tr", "pl", "nl", "sv", "fi", "no", "da", "cs"
            ],
            value="en",
            required=True
        ),
        SecretStrInput(
            name="openai_api_key",
            display_name="OpenAI API Key",
            info="Your OpenAI API key.",
            required=True
        )
    ]

    outputs = [
        Output(name="transcription", display_name="Transcribed Text", method="transcribe_audio")
    ]

    def transcribe_audio(self) -> Data:
        try:
            if not self.audio_file:
                raise ValueError("No audio file provided.")

            self.log("Transcribing audio using OpenAI Whisper")

            client = OpenAI(api_key=self.openai_api_key)

            if isinstance(self.audio_file, str):
                with open(self.audio_file, "rb") as f:
                    file_bytes = f.read()
            elif hasattr(self.audio_file, "read"):
                file_bytes = self.audio_file.read()
            elif isinstance(self.audio_file, bytes):
                file_bytes = self.audio_file
            else:
                raise TypeError("Invalid file input type.")

            if len(file_bytes) > self.MAX_FILE_SIZE_BYTES:
                raise ValueError("File exceeds 25MB limit. Please upload a smaller file.")

            buffer = io.BytesIO(file_bytes)
            buffer.name = "audio.ogg" if file_bytes[:4] == b'OggS' else "audio.wav"

            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer,
                response_format="text",
                language=self.language
            )

            return Data(data={"text": response})

        except Exception as e:
            self.status = f"Error transcribing audio: {str(e)}"
            self.log(self.status)
            return Data(data={"error": str(e)})
