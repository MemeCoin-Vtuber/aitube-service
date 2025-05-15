import os
import pygame
import live2d.v3 as live2d
import threading
import queue
import time
import json
import random
import numpy as np
from OpenGL.GL import *
from enum import Enum
from pygame.locals import *
import wave
import pyaudio

from live2d.utils.lipsync import WavHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from time import sleep

from pyht import Client
from pyht.client import TTSOptions
from smallest import Smallest
from elevenlabs import ElevenLabs

from prompts import BIO_PROMPT, LOOK_AROUND_PROMPT, GENERATE_EXPRESSION_PROMPT
from speech_generators import generate_speech_elevenlabs, generate_speech_playht, generate_speech_smallest_ai

from background import Background

class TTS_Options(Enum):
    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"
    SMALLESTAI = "smallestai"

class ExtendedWavHandler(WavHandler):
    """Extended WavHandler class with better time sync support"""
    
    def __init__(self):
        super().__init__()
        self.current_time = 0
        self.synchronized_mode = False
    
    def SetCurrentTime(self, time_seconds):
        """Set the current playback time to keep in sync with audio"""
        self.current_time = time_seconds
        self.synchronized_mode = True
        
    def Update(self):
        """Update RMS value based on current audio position"""
        if self.synchronized_mode:
            # In synchronized mode, we use the time provided by audio callback
            # rather than incrementing internally
            if self.current_time >= self.duration:
                return False
                
            # Calculate frame index based on current time
            frame_index = int(self.current_time * self.framerate)
            if frame_index < len(self.frames):
                # Get the frame data for the current position
                frame_data = self.frames[frame_index:frame_index+self.chunk_size]
                if len(frame_data) > 0:
                    # Calculate RMS for this position
                    rms_sum = 0
                    for i in range(0, len(frame_data), 2):
                        if i+1 < len(frame_data):
                            sample = int.from_bytes(frame_data[i:i+2], byteorder='little', signed=True)
                            rms_sum += sample * sample
                    
                    if len(frame_data) > 0:
                        self.rms = (rms_sum / (len(frame_data) // 2)) ** 0.5 / 32768.0
                    
                    return True
            return False
        else:
            # Use original update logic for non-synchronized mode
            return super().Update()

class Agent:
    motion_names = {}
    expression_names = []

    mouth_params = []
    vowel_params = []
    special_params = []

    audio_path = None
    current_expression = None

    def __init__(self, model_path: str, tts_option: TTS_Options, display: tuple = (1920, 1080), background=False, speak=True, virtual_mic_name="", agent_executor=None):
        
        self.display = display
        self.model_path = model_path
        self.running = True
        self.dx, self.dy = 0.0, 0.0
        self.look_dx, self.look_dy = display[0]/2, display[1]/2
        self.scale = 1.0
        self.lip_sync_multiplier = 10.0  # Increase multiplier for more sensitivity
        self.message_queue = queue.Queue()  # Queue for communication between threads
        self.current_top_clicked_part_id = None
        self.part_ids = []
        self.prompt_response = "Random movement"
        self.fps = 30
        self.frame_count = 300
        self.tts_option = tts_option
        self.speak = speak

        self.agent_executor = agent_executor
        
        self.look = {
            "left": (0, display[1]/2), 
            "right": (display[0], display[1]/2),
            "down": (display[0]/2, 0),
            "up": (display[0]/2, display[1]),
            "straight": (display[0]/2,display[1]/2)
        }

        # Mutex for audio file access
        self.audio_mutex = threading.Lock()
        self.audio_in_use = False
        self.audio_done = threading.Event()

        # Virtual mic initialization
        self.virtual_mic_name = virtual_mic_name
        self.use_virtual_mic = bool(virtual_mic_name)  # Set to True if a name is provided
        self.audio_device_id = None
        self.stream = None
        self.p = None  # PyAudio instance
        self.setup_virtual_mic()

        pygame.init()
        pygame.mixer.init()
        live2d.init()

        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Live2D Viewer")

        self.display_bg = background

        if self.display_bg:
            self.background = Background(
                os.path.join("background.png")
            )

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.model = live2d.LAppModel()
        self.wav_handler = ExtendedWavHandler()
        self.model.LoadModelJson(os.path.join(model_path))
        self.model.Resize(*display)

        # Setup TTS Models
        if tts_option == TTS_Options.ELEVENLABS:
            self.client = ElevenLabs(
                api_key=os.environ["ELEVENLABS_API_KEY"],
            )
        elif tts_option == TTS_Options.PLAYHT:
            self.client = Client(
                user_id=os.environ["PLAY_HT_USER_ID"],
                api_key=os.environ["PLAY_HT_API_KEY"]
            )
        elif tts_option == TTS_Options.SMALLESTAI:
            self.client = Smallest(
                api_key=os.environ["SMALLEST_API_KEY"],
                model=os.environ["SMALLEST_MODEL"],
                voice_id=os.environ["SMALLEST_VOICE_ID"]
            )
        else:
            raise ValueError("Invalid tts option given")

    def get_audio_duration(self, audio_file):
        """Get the duration of an audio file in seconds"""
        with wave.open(audio_file, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration

    def get_expression_names(self):
        "Extract expression names from expression directory"
        
        with open(self.model_path, "r") as file:
            data = json.load(file)
            expressions: list[dict] = data["FileReferences"].get("Expressions", [])
            expression_names = [expression["Name"] for expression in expressions]
            self.expression_names = expression_names

    def get_motion_names(self):
        "Extract motions based on group type, group = idle/moving"
        
        with open(self.model_path, "r") as file:
            data = json.load(file)
            motion_groups: dict = data["FileReferences"]["Motions"]
            for group in motion_groups.keys():
                motion_count = len(data["FileReferences"]["Motions"][group])
                self.motion_names[group] = motion_count

    def get_model_params(self):
        "Fetches facial parameters of a model, used for lypsyncing and moving the mouth"

        for i in range(self.model.GetParameterCount()):
            param = self.model.GetParameter(i)
            param_id = param.id

            if "mouth" in param_id.lower():
                self.mouth_params.append(param_id)
                print(f"Mouth param: {param_id} (min: {param.min}, max: {param.max})")

            elif param_id in ["ParamA", "ParamI", "ParamU", "ParamE", "ParamO"]:
                self.vowel_params.append(param_id)
                print(f"Vowel param: {param_id} (min: {param.min}, max: {param.max})")

            elif (
                "cheek" in param_id.lower()
                or "tongue" in param_id.lower()
                or "jaw" in param_id.lower()
            ):
                self.special_params.append(param_id)
                print(f"Special param: {param_id} (min: {param.min}, max: {param.max})")

    def setup_virtual_mic(self):
        """Set up PyAudio and identify the virtual microphone device"""
        if not self.use_virtual_mic or not self.virtual_mic_name:
            print("Virtual microphone not enabled or no name provided")
            return
            
        try:
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Print available audio devices for debugging
            print("\nAvailable audio output devices:")
            for i in range(self.p.get_device_count()):
                dev_info = self.p.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']} (maxOutputChannels: {dev_info['maxOutputChannels']})")
                # If this device name contains our virtual mic name
                if self.virtual_mic_name.lower() in dev_info['name'].lower() and dev_info['maxOutputChannels'] > 0:
                    self.audio_device_id = i
                    print(f"Found virtual microphone: {dev_info['name']} (ID: {i})")
                    break

            # If we couldn't find the device by name, log a warning
            if self.audio_device_id is None:
                print(f"\nWARNING: Could not find a device named '{self.virtual_mic_name}'")
                print("Please check that your virtual audio device is properly installed.")
                print("You can specify a device ID manually by setting audio_device_id in your code.")
                print("For now, will use the default system output.\n")
            
        except Exception as e:
            print(f"Error setting up virtual microphone: {e}")
            print("Will use default audio output instead.")

    def _calculate_rms_from_chunk(self, chunk, sample_width):
        """Calculate RMS value from a chunk of audio data"""
        if not chunk:
            return 0.0
            
        # Calculate based on bit depth
        if sample_width == 1:  # 8-bit
            max_value = 128.0
            format_char = 'b'  # signed char
        elif sample_width == 2:  # 16-bit
            max_value = 32768.0
            format_char = 'h'  # short
        elif sample_width == 3:  # 24-bit
            # Need special handling for 24-bit
            return self._calculate_rms_24bit(chunk)
        elif sample_width == 4:  # 32-bit
            max_value = 2147483648.0
            format_char = 'i'  # int
        else:
            return 0.0
        
        # Use struct to convert bytes to integers efficiently
        import struct
        
        # Calculate how many complete samples we have
        sample_count = len(chunk) // sample_width
        
        # Create format string - need to handle endianness
        format_string = f"<{sample_count}{format_char}"
        
        try:
            # Convert bytes to integers
            values = struct.unpack(format_string, chunk[:sample_count*sample_width])
            
            # Calculate RMS
            sum_squares = sum(v*v for v in values)
            rms = (sum_squares / sample_count) ** 0.5 / max_value
            
            # Cap RMS at 1.0 
            return min(rms, 1.0)
        except:
            # Fallback method if struct unpacking fails
            sum_squares = 0
            count = 0
            
            for i in range(0, len(chunk), sample_width):
                if i + sample_width <= len(chunk):
                    # Extract sample based on bit depth
                    if sample_width == 1:
                        sample = chunk[i]
                        if sample > 127:
                            sample = sample - 256
                    else:
                        sample = int.from_bytes(chunk[i:i+sample_width], byteorder='little', signed=True)
                    
                    sum_squares += (sample / max_value) ** 2
                    count += 1
            
            if count == 0:
                return 0.0
                
            return min((sum_squares / count) ** 0.5, 1.0)


    def play_audio_to_virtual_mic(self, audio_file):
        """Play audio file through the virtual mic and calculate RMS in real-time"""
        try:
            # Load the audio file
            with wave.open(audio_file, 'rb') as wf:
                # Get audio parameters
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                
                # Convert sample_width to PyAudio format
                format_mapping = {
                    1: pyaudio.paInt8,
                    2: pyaudio.paInt16,
                    3: pyaudio.paInt24,
                    4: pyaudio.paInt32
                }
                audio_format = format_mapping.get(sample_width, pyaudio.paInt16)
                
                # Create a callback function for processing audio chunks
                def callback(in_data, frame_count, time_info, status):
                    nonlocal audio_data, position
                    
                    if position >= len(audio_data):
                        # End of audio
                        self.audio_in_use = False
                        self.audio_done.set()
                        return (None, pyaudio.paComplete)
                    
                    # Calculate bytes per frame
                    bytes_per_frame = sample_width * channels
                    bytes_to_read = frame_count * bytes_per_frame
                    
                    # Get current chunk
                    chunk = audio_data[position:position + bytes_to_read]
                    position += bytes_to_read
                    
                    # Calculate RMS directly from the audio chunk
                    rms = self._calculate_rms_from_chunk(chunk, sample_width)
                    
                    # Store the RMS value for use in the main thread
                    self.current_rms = (rms * 10) / 2
                    
                    # If we need padding
                    if len(chunk) < bytes_to_read:
                        chunk += b'\x00' * (bytes_to_read - len(chunk))
                    
                    return (chunk, pyaudio.paContinue)
                
                # Initialize position counter
                position = 0
                self.current_rms = 0.0
                self.audio_in_use = True
                
                # Create and start the stream
                stream = self.p.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    output_device_index=self.audio_device_id,
                    frames_per_buffer=1024,  # Adjust buffer size as needed
                    stream_callback=callback
                )
                
                # Start the stream
                stream.start_stream()
                self.stream = stream
                
                return True
                
        except Exception as e:
            print(f"Error playing audio with RMS calculation: {e}")
            import traceback
            traceback.print_exc()
            self.audio_in_use = False
            self.audio_done.set()
            return False

    

    def generate_speech(self, text):
        # Create a temporary filename to avoid conflicts
        temp_filename = f"output_temp.wav"
        
        if self.tts_option == TTS_Options.ELEVENLABS:
            generate_speech_elevenlabs(
                self.client, 
                text,
                os.environ["ELEVENLABS_VOICE_ID"],
                os.environ["ELEVENLABS_MODEL_ID"] 
            )
        elif self.tts_option == TTS_Options.PLAYHT:
            generate_speech_playht(
                self.client,
                text,
                os.environ["PLAYHT_VOICE_MANIFEST_URL"]
            )
        elif self.tts_option == TTS_Options.SMALLESTAI:
            generate_speech_smallest_ai(
                self.client,
                text
            )
        else:
            raise ValueError("Invalid TTS option passed")
        
        # Acquire mutex before renaming file
        with self.audio_mutex:
            # If there's an old output file, remove it
            if os.path.exists("output.wav"):
                try:
                    os.remove("output.wav")
                except:
                    pass
            
            # Rename temp file to final filename
            try:
                os.rename(temp_filename, "output.wav")
            except Exception as e:
                print(f"Error renaming audio file: {e}")
                # If rename fails, at least return the temp file
                return temp_filename
                
        return "output.wav"

    def llm_worker(self):
        """Worker thread to generate LLM content and speech"""

        generate_expression_chain = GENERATE_EXPRESSION_PROMPT | self.agent_executor 
        generate_response_chain = BIO_PROMPT | self.agent_executor

        while self.running:
            try:
                # Wait for previous audio to finish playing
                if self.audio_in_use:
                    print("LLM thread: Waiting for audio to finish playing...")
                    self.audio_done.wait(timeout=10)  # Wait with timeout
                    self.audio_done.clear()
                
                print("LLM thread: Generating content...")
                content = self.agent_executor.predict(input = BIO_PROMPT.format(expressions=self.expression_names))
                self.prompt_response = content

                # Generate expression
                print("LLM thread: Generating expression...")
                expression = self.agent_executor.predict(input = GENERATE_EXPRESSION_PROMPT.format(content=content, expression_names=self.expression_names))
                print(f"LLM thread: Expression generated: {expression}")
                
                # Generate speech
                print("LLM thread: Generating speech...")
                audio_file = self.generate_speech(content)
                print(f"LLM thread: Speech generated to {audio_file}")
                
                # Put message in queue for main thread to process
                print("LLM thread: Putting message in queue...")
                self.message_queue.put({
                    "content": content,
                    "expression": expression,
                    "audio_file": audio_file,
                    "timestamp": time.time()
                })
                print("LLM thread: Message in queue, sleeping for 3 seconds...")
                
                # Sleep with timeout check to avoid getting stuck
                start_time = time.time()
                while time.time() - start_time < 3 and self.running:  # 3 seconds
                    sleep(0.1)  # Short sleep to allow for cleaner thread exit
                
                sleep(10)
                print("LLM thread: Woke up, starting next iteration")
            except Exception as e:
                print(f"Error in LLM worker: {e}")
                # Sleep with shorter timeout for error recovery
                sleep(5)

    def idle_motion_worker(self):
        groups = list(self.motion_names.keys())

        while True:
            try:
                # Use a try-except block to handle motion errors
                if "Idle" in self.motion_names and self.motion_names["Idle"] > 0:
                    motionIdx = random.randint(0, self.motion_names["Idle"] - 1)
                    print(f"Starting Idle motion {motionIdx}")
                    self.model.StartMotion("Idle", motionIdx, 1)
                else:
                    # Fallback if no Idle motions are available
                    for group in groups:
                        if self.motion_names[group] > 0:
                            motionIdx = random.randint(0, self.motion_names[group] - 1)
                            print(f"Starting {group} motion {motionIdx}")
                            self.model.StartMotion(group, motionIdx, 1)
                            break
            except Exception as e:
                print(f"Error in idle_motion_worker: {e}")
                
            # Add a slight randomization to the sleep time
            sleep(random.randint(8, 20))

    def look_around_worker(self):
        while True:
            choices = ["straight"] * 6 + ["left", "right", "up", "down"]  # 60% straight, 10% others
            selected = random.choice(choices)
            self.look_dx, self.look_dy = self.look[selected]
            print("Look selected: ", selected)
            sleep(5)

    def run_agent(self):
        """Main method that runs everything"""
        print("Starting agent....")

        self.get_expression_names()
        self.get_motion_names()
        self.get_model_params()
        
        # Start LLM thread
        self.running = True
        llm_thread = threading.Thread(target=self.llm_worker)
        llm_thread.daemon = True  # Make thread daemon so it exits when main thread exits

        if self.speak:
            llm_thread.start()

        expression_thread = threading.Thread(target=self.look_around_worker)
        expression_thread.daemon = True
        expression_thread.start()

        motion_thread = threading.Thread(target=self.idle_motion_worker)
        motion_thread.daemon = True
        motion_thread.start()
        
        print("Main thread: LLM worker thread started")
        print("Main thread: Starting video loop")
        
        # Run the main loop in the main thread
        try:
            self.run_video()
        except Exception as e:
            print(f"Main thread: Error in video loop: {e}")
        finally:
            # Cleanup
            print("Main thread: Shutting down")
            self.running = False
            
            # Signal any waiting threads
            self.audio_done.set()
            
            # Wait for worker thread to finish any current work (with timeout)
            print("Main thread: Waiting for worker thread to exit")
            start_time = time.time()
            while llm_thread.is_alive() and time.time() - start_time < 5:  # 5 second timeout
                sleep(0.1)

            while expression_thread.is_alive() and time.time() - start_time < 5:  # 5 second timeout
                sleep(0.1)

            while motion_thread.is_alive() and time.time() - start_time < 5:  # 5 second timeout
                sleep(0.1)
                
            print("Main thread: Cleaning up PyGame and Live2D")
            pygame.quit()
            live2d.dispose()
            print("Main thread: Shutdown complete")

    def run_video(self):
        """Main video loop - must run in main thread"""

        clock = pygame.time.Clock()

        while self.running:
            frame_start = time.time()

            # Process PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Check for messages from the LLM thread
            try:
                if not self.message_queue.empty() and not self.audio_in_use:
                    print("Main thread: Found message in queue")
                    message = self.message_queue.get_nowait()
                    
                    # Apply expression and play audio
                    self.current_expression = message["expression"]
                    self.audio_path = message["audio_file"]
                    
                    print(f"Main thread: Processing message with expression: {self.current_expression}")
                    
                    # Handle in main thread
                    self.model.SetExpression(self.current_expression)

                    # Acquire mutex before accessing audio file
                    with self.audio_mutex:
                        try:
                            if self.use_virtual_mic and self.audio_device_id is not None:
                                # Use virtual mic with synchronized streaming
                                self.audio_in_use = True
                                if self.play_audio_to_virtual_mic(self.audio_path):
                                    # Fallback to pygame if virtual mic fails
                                    pygame.mixer.music.load(self.audio_path)
                                    pygame.mixer.music.play()
                                    self.wav_handler.Start(self.audio_path)
                                    pygame.mixer.music.set_volume(0.0)
                                    print(f"Main thread: Playing audio via virtual mic {self.audio_path}")
                                
                            else:
                                # Use pygame mixer
                                pygame.mixer.music.load(self.audio_path)
                                self.audio_in_use = True
                                pygame.mixer.music.play()
                                self.wav_handler.Start(self.audio_path)
                                print(f"Main thread: Playing audio via pygame {self.audio_path}")
                        except Exception as e:
                            print(f"Main thread: Error playing audio: {e}")
                            self.audio_in_use = False
                    
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Main thread: Error processing message: {e}")

            # Update the model
            self.model.Update()

            # Handle lip sync - NOTE: We now handle this differently for virtual mic
            # as lip sync is managed by the audio callback
            if self.audio_in_use:
                if self.use_virtual_mic and hasattr(self, 'stream') and self.stream and self.stream.is_active():
                    # For virtual mic, lip sync is managed by the audio callback
                    # We just need to apply the current RMS value that's updated there
                    self._apply_mouth_movement(self.current_rms)  # Use random value for testing
                    print(f"Main thread: RMS value: {self.current_rms}")
                elif pygame.mixer.music.get_busy() and self.wav_handler.Update():
                    # For pygame, we update the wav handler and apply mouth movement
                    rms = self.wav_handler.GetRms()
                    self._apply_mouth_movement(rms)
                    print(f"Main thread: RMS value: {rms}")

                elif not pygame.mixer.music.get_busy():
                    # Reset parameters if pygame audio stops
                    self._reset_mouth_parameters()
            
            # Check if audio finished playing
            if self.audio_in_use and self.audio_done.is_set():
                # Audio finished playing
                print("Main thread: Audio finished playing")
                self.audio_in_use = False
                self.audio_done.clear()  # Reset the event
                
                # Close stream if open
                if hasattr(self, 'stream') and self.stream and self.stream.is_active():
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                
                self.model.SetExpression("normal")
                self._reset_mouth_parameters()

            self.model.SetOffset(self.dx, self.dy)
            self.model.SetScale(self.scale)
            self.model.Drag(self.look_dx, self.look_dy)
            
            # Change alpha to 1.0 instead of 0.0 (not transparent)
            live2d.clearBuffer(0.0, 0.0, 0.0, 1.0)
            
            if self.display_bg:
                self.background.Draw()
                self.model.Update()
            self.model.Draw()

            pygame.display.flip()

            # FPS limiting
            frame_end = time.time()
            frame_time = frame_end - frame_start
            target_frame_time = 1.0 / self.fps
            
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)

            clock.tick(60)

    def _apply_mouth_movement(self, rms):
        """Apply mouth movement based on RMS value - extracted to separate method for reuse"""
        # Cap the RMS value to prevent extreme mouth movements
        max_rms = 0.3
        if rms > max_rms:
            rms = max_rms

        for param_id in self.mouth_params:
            try:
                if "openy" in param_id.lower():
                    self.model.SetParameterValue(param_id, rms * self.lip_sync_multiplier)
                elif "form" in param_id.lower():
                    self.model.SetParameterValue(param_id, rms * 0.5)
            except Exception as e:
                print(f"Error setting mouth parameter {param_id}: {e}")

        if self.vowel_params:
            try:
                # Reset all vowel params first
                for param_id in self.vowel_params:
                    self.model.SetParameterValue(param_id, 0.0)
                    
                # Set just one vowel parameter based on RMS range
                if rms > 0.2:
                    # For loud sounds, use "A"
                    if "ParamA" in self.vowel_params:
                        self.model.SetParameterValue("ParamA", rms * 3.0)
                elif 0.1 < rms <= 0.2:
                    # For medium sounds, use "O"
                    if "ParamO" in self.vowel_params:
                        self.model.SetParameterValue("ParamO", rms * 2.0)
                elif 0.05 < rms <= 0.1:
                    # For softer sounds, alternate between "E" and "U"
                    if random.random() > 0.5 and "ParamE" in self.vowel_params:
                        self.model.SetParameterValue("ParamE", rms * 2.0)
                    elif "ParamU" in self.vowel_params:
                        self.model.SetParameterValue("ParamU", rms * 2.0)
                else:
                    # For very soft sounds, use "I"
                    if "ParamI" in self.vowel_params:
                        self.model.SetParameterValue("ParamI", rms * 2.0)
            except Exception as e:
                print(f"Error setting vowel parameters: {e}")

    def _reset_mouth_parameters(self):
        """Reset all mouth parameters to avoid them getting stuck"""
        # Reset all mouth parameters
        for param_id in self.mouth_params:
            try:
                self.model.SetParameterValue(param_id, 0.0)
            except Exception as e:
                print(f"Error resetting mouth parameter {param_id}: {e}")
        
        # Reset all vowel parameters
        for param_id in self.vowel_params:
            try:
                self.model.SetParameterValue(param_id, 0.0)
            except Exception as e:
                print(f"Error resetting vowel parameter {param_id}: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")
    os.environ["PLAY_HT_USER_ID"] = os.getenv("PLAY_HT_USER_ID")
    os.environ["PLAY_HT_API_KEY"] = os.getenv("PLAY_HT_API_KEY")
    os.environ["SMALLEST_API_KEY"] = os.getenv("SMALLEST_API_KEY")
    os.environ["SMALLEST_MODEL"] = os.getenv("SMALLEST_MODEL")
    os.environ["SMALLEST_VOICE_ID"] = os.getenv("SMALLEST_VOICE_ID")


    llm = ChatOpenAI(temperature=0)

    # 2. Set up memory (stores chat history)
    memory = ConversationBufferMemory(return_messages=True)

    agent_executor = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    tts_option = TTS_Options(os.getenv("TTS_OPTION"))
    agt = Agent("Resources/Mao/Mao.model3.json", tts_option, background=False, speak=True, virtual_mic_name="BlackHole 64ch", agent_executor=agent_executor)
    agt.run_agent()