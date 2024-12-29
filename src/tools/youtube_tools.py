import os
import requests
from langchain.tools import BaseTool

class YouTubeTranscriptTool(BaseTool):
    name = "youtube_transcript_tool"
    description = "Extracts transcript from a YouTube video URL using Supadata API"

    def _run(self, url: str) -> str:
        api_key = os.getenv('SUPADATA_API_KEY')
        headers = {
            'x-api-key': api_key
        }
        
        response = requests.get(
            'https://api.supadata.ai/v1/youtube/transcript',
            params={'url': url, 'text': True},
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()['content']
        else:
            return f"Error: {response.status_code} - {response.text}"

    def _arun(self, url: str):
        raise NotImplementedError("Async not implemented") 