import os
import pygame
import live2d.v3 as live2d
import threading
import queue
import time

from pygame.locals import *

from live2d.utils import log
from live2d.utils.lipsync import WavHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from time import sleep

from pyht import Client
from pyht.client import TTSOptions
from smallest import Smallest

class Agent:

    model = None
    motion_names = []
    expression_names = []

    mouth_params = []
    vowel_params = []
    special_params = []

    audio_path = None
    current_expression = None


    def __init__(self, model_path: str, model_file: str, display: list = (1000, 1700)):
        
        self.display = display
        self.model_path = model_path
        self.running = True
        self.dx, self.dy = 0.0, 0.0
        self.scale = 1.0
        self.lip_sync_multiplier = 10.0  # Increase multiplier for more sensitivity
        self.message_queue = queue.Queue()  # Queue for communication between threads
        self.current_top_clicked_part_id = None
        self.part_ids = []
        
        # Mutex for audio file access
        self.audio_mutex = threading.Lock()
        self.audio_in_use = False
        self.audio_done = threading.Event()

        pygame.init()
        pygame.mixer.init()
        live2d.init()

        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Live2D Viewer")

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.model = live2d.LAppModel()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"])
        self.wav_handler = WavHandler()
        # self.playht_client = Client(
        #     user_id=os.environ["PLAY_HT_USER_ID"],
        #     api_key=os.environ["PLAY_HT_API_KEY"],
        # )
        self.smallest_ai_client = Smallest(api_key=os.environ["SMALLEST_API_KEY"])


        self.model.LoadModelJson(os.path.join(model_path, model_file))
        self.model.Resize(*display)


    def get_expression_names(self, path: str):
        expression_file_names = os.listdir(os.path.join(path, "expressions"))
        expression_names = [expression.split(".")[0] for expression in expression_file_names]
        self.expression_names = expression_names

    def get_motion_names(self, path: str):
        motion_file_names = os.listdir(os.path.join(path, "motions"))
        motion_names = [expression.split(".")[0] for expression in motion_file_names]
        self.motion_names = motion_names

    def get_model_params(self):
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
        
        self.smallest_ai_client.synthesize(
            text=text,
            save_as=temp_filename
        )
        
        # Acquire mutex before renaming file
        with self.audio_mutex:
            # If there's an old output file, remove it
            if os.path.exists("output_jenn.wav"):
                try:
                    os.remove("output_jenn.wav")
                except:
                    pass
            
            # Rename temp file to final filename
            try:
                os.rename(temp_filename, "output_jenn.wav")
            except Exception as e:
                print(f"Error renaming audio file: {e}")
                # If rename fails, at least return the temp file
                return temp_filename
                
        return "output_jenn.wav"

    def generate_expression(self, content):
        response = self.llm.invoke(f"""
            given below are the list of expression names, based on given text below output any one expression that
            fits bets for the text emotion. output action with no space nothing and exact expression name nothing else.
            try to use all the expression and not repeat.
            Expression:
            {self.expression_names}   

            Text:
            {content}

            output relevent expression based on the given text
        """)

        return response.content

    def llm_worker(self):
        """Worker thread to generate LLM content and speech"""
        while self.running:
            try:
                # Wait for previous audio to finish playing
                if self.audio_in_use:
                    print("LLM thread: Waiting for audio to finish playing...")
                    self.audio_done.wait(timeout=10)  # Wait with timeout
                    self.audio_done.clear()
                
                print("======")
                print("LLM thread: Generating content...")
                response = self.llm.invoke("""
                    You are a beautiful crypto anime character, generate normal talk of short length max 50-100 words, 
                    only give me the text nothing else
                """)
                content = response.content
                print(f"LLM thread: Content generated: {content[:30]}...")

                # Generate expression
                print("LLM thread: Generating expression...")
                expression = self.generate_expression(content)
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
                    sleep(5)  # Short sleep to allow for cleaner thread exit
                
                print("LLM thread: Woke up, starting next iteration")
            except Exception as e:
                print(f"Error in LLM worker: {e}")
                # Sleep with shorter timeout for error recovery
                sleep(2)

    def run_agent(self):
        """Main method that runs everything"""
        print("Starting agent....")
        
        # Start LLM thread
        self.running = True
        llm_thread = threading.Thread(target=self.llm_worker)
        llm_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        llm_thread.start()
        
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
                
            print("Main thread: Cleaning up PyGame and Live2D")
            pygame.quit()
            live2d.dispose()
            print("Main thread: Shutdown complete")

    def run_video(self):
        """Main video loop - must run in main thread"""
        self.get_expression_names(self.model_path)
        self.get_motion_names(self.model_path)
        self.get_model_params()

        clock = pygame.time.Clock()

        while self.running:
            # Process PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()

                    hit_parts = self.model.HitPart(x, y, False)
                    if hit_parts:
                        self.current_top_clicked_part_id = hit_parts[0]

                    if event.button == 1:
                        self.model.SetExpression("exp_06")
                        self.model.StartRandomMotion("TapBody", 3)

                    if event.button == 3:
                        self.model.SetExpression("exp_06")
                        self.model.StartRandomMotion("TapBody", 3)
                        if self.audio_path:
                            with self.audio_mutex:
                                pygame.mixer.music.load(self.audio_path)
                                pygame.mixer.music.play()
                                self.wav_handler.Start(self.audio_path)

                if event.type == pygame.MOUSEMOTION:
                    self.model.Drag(*pygame.mouse.get_pos())

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.dx -= 0.1
                    elif event.key == pygame.K_RIGHT:
                        self.dx += 0.1
                    elif event.key == pygame.K_UP:
                        self.dy += 0.1
                    elif event.key == pygame.K_DOWN:
                        self.dy -= 0.1
                    elif event.key == pygame.K_i:
                        self.scale += 0.1
                    elif event.key == pygame.K_u:
                        self.scale -= 0.1

                    elif event.key == pygame.K_q:
                        self.model.StartMotion("TapBody", 1, 3)
                    elif event.key == pygame.K_w:
                        self.model.StartMotion("TapBody", 2, 3)
                    elif event.key == pygame.K_e:
                        self.model.StartMotion("TapBody", 3, 3)
                    elif event.key == pygame.K_r:
                        self.model.StartMotion("TapBody", 4, 3)
                    elif event.key == pygame.K_t:
                        self.model.StartMotion("TapBody", 5, 3)
                    elif event.key == pygame.K_y:
                        self.model.StartMotion("TapBody", 6, 3)
                    elif event.key == pygame.K_a:
                        self.model.StartMotion("TapBody", 7, 3)
                    elif event.key == pygame.K_s:
                        self.model.StartMotion("TapBody", 8, 3)

                    elif event.key == pygame.K_r:
                        self.model.StopAllMotions()
                        self.model.ResetPose()
                    elif event.key == pygame.K_e:
                        self.model.ResetExpression()

                    elif event.key == pygame.K_l:
                        if self.audio_path:
                            with self.audio_mutex:
                                self.wav_handler.Start(self.audio_path)

                    elif event.key == pygame.K_SPACE:
                        blink = not self.model.GetAutoBlinkEnable()
                        breath = not self.model.GetAutoBreathEnable()
                        self.model.SetAutoBlinkEnable(blink)
                        self.model.SetAutoBreathEnable(breath)

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

            if self.current_top_clicked_part_id is not None:
                try:
                    idx = self.part_ids.index(self.current_top_clicked_part_id)
                    self.model.SetPartOpacity(idx, 0.7)
                    self.model.SetPartMultiplyColor(idx, 0.0, 0.0, 1.0, 0.9)
                except:
                    pass

            self.model.SetOffset(self.dx, self.dy)
            self.model.SetScale(self.scale)
            live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
            self.model.Draw()

            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")
    os.environ["PLAY_HT_USER_ID"] = os.getenv("PLAY_HT_USER_ID")
    os.environ["PLAY_HT_API_KEY"] = os.getenv("PLAY_HT_API_KEY")
    os.environ["SMALLEST_API_KEY"] = os.getenv("SMALLEST_API_KEY")

    

    agt = Agent("Resources/Mao", "Mao.model3.json")
    agt.run_agent()