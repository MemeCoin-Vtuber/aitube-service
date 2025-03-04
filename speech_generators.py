from smallest import Smallest
from elevenlabs import ElevenLabs, save

from pyht import Client
from pyht.client import TTSOptions


def generate_speech_smallest_ai(client: Smallest, text: str):
        
    temp_filename = f"output_temp.wav"
    
    client.synthesize(
        text=text,
        save_as=temp_filename
    )
    
    return temp_filename


def generate_speech_playht(client: Client, text: str, voice_manifest_url: str = "s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json"):
        
    temp_filename = f"output_temp.wav"

    options = TTSOptions(voice=voice_manifest_url)

    with open(temp_filename, "wb") as audio_file:
        for chunk in client.tts(text, options, voice_engine = 'PlayDialog', protocol="http"):
            audio_file.write(chunk)
            
    return temp_filename

def generate_speech_elevenlabs(client: ElevenLabs, text: str, voice_id: str, model_id: str):
      
    temp_filename = f"output_temp.wav"


    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="pcm_22050",
    )

    save(audio, temp_filename)
