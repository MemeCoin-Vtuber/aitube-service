import os
import pygame
import live2d.v3 as live2d
import threading
import queue
import time
import json
import random
import cv2
import numpy as np
from  OpenGL.GL import *
from enum import Enum
from pygame.locals import *
import wave
import subprocess

from live2d.utils import log
from live2d.utils.lipsync import WavHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from time import sleep

from pyht import Client
from pyht.client import TTSOptions
from smallest import Smallest
from elevenlabs import ElevenLabs

from prompts import BIO_PROMPT, LOOK_AROUND_PROMPT, GENERATE_EXPRESSION_PROMPT
from speech_generators import generate_speech_elevenlabs, generate_speech_playht, generate_speech_smallest_ai



class TTS_Options(Enum):

    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"
    SMALLESTAI = "smallestai"



def capture_frame(width, height):
    """Capture the current OpenGL frame"""
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)  # OpenGL has origin at bottom left
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

class Agent:

    motion_names = {}
    expression_names = []

    mouth_params = []
    vowel_params = []
    special_params = []

    audio_path = None
    current_expression = None


    def __init__(self, model_path: str, tts_option: TTS_Options, youtube_key: str, display: tuple = (700, 700), speak=True):
        
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
        self.youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{youtube_key}"
        
        self.look = {
            "left": (0, display[1]/2), 
            "right": (display[0], display[1]/2),
            "down": (display[0]/2, 0),
            "up": (display[0]/2, display[1]),
            "straight": (display[0]/2,display[1]/2)
        }

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            
            # Video Input
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.display[0]}x{self.display[1]}',
            '-r', str(self.fps),
            '-i', '-',  # Read raw video from stdin

            '-f', 'lavfi',
            '-i', 'anullsrc=r=44100:cl=stereo',
            
            # Video Encoding
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-b:v', '2500k',
            '-maxrate', '2500k',
            '-bufsize', '5000k',
            '-g', str(self.fps * 2),
            
            # Audio Encoding
            '-c:a', 'aac',
            '-b:a', '128k',
            
            # Output Format (FLV for YouTube)
            '-f', 'flv',
            self.youtube_url
        ]

        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # Mutex for audio file access
        self.audio_mutex = threading.Lock()
        self.audio_in_use = False
        self.audio_done = threading.Event()

        pygame.init()
        pygame.mixer.init()
        live2d.init()

        self.surface = pygame.Surface(display)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter("output.mp4", fourcc, self.fps, display)

        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Live2D Viewer")

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.model = live2d.LAppModel()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"])
        self.wav_handler = WavHandler()

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


    def generate_speech(self, text):
        # Create a temporary filename to avoid conflicts
        # temp_filename = f"output_temp_{int(time.time())}.wav"
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

        generate_expression_chain = GENERATE_EXPRESSION_PROMPT | self.llm 
        generate_response_chain = BIO_PROMPT | self.llm

        while self.running:
            try:
                # Wait for previous audio to finish playing
                if self.audio_in_use:
                    print("LLM thread: Waiting for audio to finish playing...")
                    self.audio_done.wait(timeout=10)  # Wait with timeout
                    self.audio_done.clear()
                
                print("LLM thread: Generating content...")

                response = generate_response_chain.invoke({"expressions": self.expression_names})
                content = response.content
                self.prompt_response = content

                print(f"LLM thread: Content generated: {content[:30]}...")

                # Generate expression
                print("LLM thread: Generating expression...")
                response = generate_expression_chain.invoke({"expression_names": self.expression_names, "content": content})
                expression = response.content
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
                    sleep(10)  # Short sleep to allow for cleaner thread exit
                
                print("LLM thread: Woke up, starting next iteration")
            except Exception as e:
                print(f"Error in LLM worker: {e}")
                # Sleep with shorter timeout for error recovery
                sleep(2)

    def idle_motion_worker(self):

        groups = list(self.motion_names.keys())

        while True:
            
            selected = random.choices(groups)

            self.model.StartRandomMotion(selected[0], 3)

            sleep(random.randint(8,20))

    def look_around_worker(self):

        while True:
            # print("Look around started")
            # response = self.llm.invoke(f"""
            #     Given {self.display} which is size of the total screen,
            #     Now in center of the screen we have a Animated character 
            #     which is talking and needs to look around.

            #     Your task is to generate a point where it will look currently 

            #     eg: [100,200]
            #     this examples shows the character will be looking at this point
            #     generate this head movement based on given content it will be speaking.
                
            #     Don't generate random points currently the head is looking at 
            #     [{self.look_dx}, {self.look_dy}], so generate new point based on this and 
            #     it should not look random.
                
            #     if the Content doesnt make sense then generate what you think is good.
            #     your response should not not be a program and no comments

            #     Content:
            #     {self.prompt_response}
            # """)

            # try:
            #     print(response.content)
            #     movement: list[2] = json.loads(response.content)
            #     self.look_dx = movement[0]
            #     self.look_dy = movement[1]

            # except Exception as e:
            #     print(e)
            #     pass

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
            # Process PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False


                # if event.type == pygame.MOUSEMOTION:
                #     self.model.Drag(*pygame.mouse.get_pos())

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
                            pygame.mixer.music.load(self.audio_path)
                            self.audio_in_use = True
                            pygame.mixer.music.play()
                            self.wav_handler.Start(self.audio_path)
                            print(f"Main thread: Playing audio {self.audio_path}")
                        except Exception as e:
                            print(f"Main thread: Error playing audio: {e}")
                            self.audio_in_use = False
                    
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Main thread: Error processing message: {e}")

            # Update the model
            self.model.Update()

            # Handle lip sync
            if pygame.mixer.music.get_busy() and self.wav_handler.Update():
                rms = self.wav_handler.GetRms()

                for param_id in self.mouth_params:
                    try:
                        if "openy" in param_id.lower():
                            self.model.AddParameterValue(param_id, rms * self.lip_sync_multiplier)
                        elif "form" in param_id.lower():
                            self.model.AddParameterValue(param_id, rms * 0.5)
                    except Exception as e:
                        pass

                if self.vowel_params:
                    try:
                        if "ParamA" in self.vowel_params:
                            self.model.AddParameterValue(
                                "ParamA", rms * 3.0 if rms > 0.05 else 0
                            )
                        if "ParamO" in self.vowel_params:
                            self.model.AddParameterValue(
                                "ParamO", rms * 2.0 if 0.04 < rms < 0.15 else 0
                            )
                        if "ParamI" in self.vowel_params:
                            self.model.AddParameterValue(
                                "ParamI", rms * 1.0 if rms < 0.06 else 0
                            )
                        if "ParamU" in self.vowel_params:
                            self.model.AddParameterValue(
                                "ParamU", rms * 1.5 if 0.03 < rms < 0.1 else 0
                            )
                        if "ParamE" in self.vowel_params:
                            self.model.AddParameterValue(
                                "ParamE", rms * 1.0 if 0.03 < rms < 0.08 else 0
                            )
                    except Exception as e:
                        pass

                print(f"RMS: {rms:.3f}")
            
            # Check if audio finished playing
            if self.audio_in_use and not pygame.mixer.music.get_busy():
                # Audio finished playing
                print("Main thread: Audio finished playing")
                self.audio_in_use = False
                self.audio_done.set()  # Signal that audio is done
                self.model.SetExpression("normal")

            self.model.SetOffset(self.dx, self.dy)
            self.model.SetScale(self.scale)
            # self.model.HitPart(100, 200, False)
            self.model.Drag(self.look_dx, self.look_dy)
            live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
            self.model.Draw()

            pygame.display.flip()

            frame = capture_frame(self.display[0], self.display[1])
            self.ffmpeg_process.stdin.write(frame.tobytes())


            # self.video_writer.write(frame)

            # self.frame_count += 1
            # if self.frame_count % self.fps == 0:
            #     seconds = self.frame_count // self.fps
            #     print(f"Recorded {seconds} seconds of video ({self.frame_count} frames)")

            # if self.frame_count > 1500:
            #     self.video_writer.release()

            clock.tick(60)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")
    os.environ["PLAY_HT_USER_ID"] = os.getenv("PLAY_HT_USER_ID")
    os.environ["PLAY_HT_API_KEY"] = os.getenv("PLAY_HT_API_KEY")
    os.environ["SMALLEST_API_KEY"] = os.getenv("SMALLEST_API_KEY")

    os.environ["SMALLEST_API_KEY"] = os.getenv("SMALLEST_API_KEY")
    os.environ["SMALLEST_API_KEY"] = os.getenv("SMALLEST_API_KEY")
    os.environ["YOUTUBE_STREAM_KEY"] = os.getenv("YOUTUBE_STREAM_KEY")

    tts_option = TTS_Options(os.getenv("TTS_OPTION"))
    agt = Agent("Resources/Mao/Mao.model3.json",tts_option, os.environ["YOUTUBE_STREAM_KEY"],speak=False)
    agt.run_agent()