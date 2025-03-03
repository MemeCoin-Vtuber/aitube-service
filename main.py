import os
import pygame
from pygame.locals import *
import live2d.v3 as live2d
from live2d.utils import log
from live2d.utils.lipsync import WavHandler
from OpenGL.GL import *
import numpy as np
import subprocess
import time
import tempfile
import shutil
import wave
import json

width, height = 1000, 1400
fps = 30
temp_dir = tempfile.mkdtemp()
frames_dir = os.path.join(temp_dir, "frames")
audio_dir = os.path.join(temp_dir, "audio")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)
output_filename = "expression_showcase.mp4"
audio_config_file = os.path.join(temp_dir, "audio_config.json")

def get_audio_duration(audio_file):
    """Get the duration of an audio file in seconds"""
    with wave.open(audio_file, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def capture_frame():
    """Capture the current OpenGL frame"""
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)  # OpenGL has origin at bottom left
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

def create_video_with_multiple_audio(frames_dir, audio_config, output_filename, fps):
    """Create video with multiple audio sources at specific times"""
    try:
        # First create a video without audio
        temp_video = os.path.join(temp_dir, "temp_video_no_audio.mp4")
        cmd = [
            'ffmpeg', 
            '-y',
            '-framerate', str(fps), 
            '-i', os.path.join(frames_dir, 'frame_%05d.png'), 
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            temp_video
        ]
        subprocess.run(cmd, check=True)
        
        # Create a file with timing information for complex filtering
        filter_file = os.path.join(temp_dir, "filter_complex.txt")
        
        # Start building the complex filter
        filter_parts = []
        input_count = 1  # Start after the video input
        
        # For each audio segment
        for idx, audio_segment in enumerate(audio_config):
            segment_file = audio_segment['file']
            start_time = audio_segment['start_time']
            
            # Add input for this segment
            filter_parts.append(f"[{input_count}:a]adelay={int(start_time*1000)}|{int(start_time*1000)}[a{idx}]")
            input_count += 1
        
        # Mix all audio segments
        if audio_config:
            # Create the mix command
            mix_inputs = ''.join(f'[a{i}]' for i in range(len(audio_config)))
            filter_parts.append(f"{mix_inputs}amix=inputs={len(audio_config)}:duration=longest[aout]")
            
            # Write to filter file
            with open(filter_file, 'w') as f:
                f.write(";\n".join(filter_parts))
            
            # Build input arguments for all audio files
            input_args = []
            for segment in audio_config:
                input_args.extend(['-i', segment['file']])
            
            # Final command to combine video with all audio segments
            cmd = [
                'ffmpeg',
                '-y',
                '-i', temp_video
            ] + input_args + [
                '-filter_complex_script', filter_file,
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                output_filename
            ]
            subprocess.run(cmd, check=True)
            print(f"Successfully created video with multiple audio tracks: {output_filename}")
        else:
            # No audio, just copy the video
            shutil.copy(temp_video, output_filename)
            print(f"Created video with no audio: {output_filename}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg processing: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    pygame.init()
    pygame.mixer.init()
    live2d.init()

    # Import cv2 here to avoid any startup issues
    global cv2
    import cv2

    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Live2D Viewer")

    if live2d.LIVE2D_VERSION == 3:
        live2d.glewInit()

    model = live2d.LAppModel()
    model.LoadModelJson(os.path.join("Resources/Mao/Mao.model3.json"))
    model.Resize(*display)

    running = True
    dx, dy = 0.0, 0.0
    scale = 1.0
    audio_path = os.path.join("Resources/audio1.wav")
    audio_duration = get_audio_duration(audio_path)
    print(f"Audio duration: {audio_duration:.2f} seconds")

    wav_handler = WavHandler()
    lip_sync_multiplier = 10.0
    lip_sync_active = False

    # Keep track of audio segments and their start times
    audio_segments = []
    current_audio_segment = None

    part_ids = model.GetPartIds()
    current_top_clicked_part_id = None

    mouth_params = []
    vowel_params = []
    special_params = []

    for i in range(model.GetParameterCount()):
        param = model.GetParameter(i)
        param_id = param.id

        if "mouth" in param_id.lower():
            mouth_params.append(param_id)
            print(f"Mouth param: {param_id} (min: {param.min}, max: {param.max})")
        elif param_id in ["ParamA", "ParamI", "ParamU", "ParamE", "ParamO"]:
            vowel_params.append(param_id)
            print(f"Vowel param: {param_id} (min: {param.min}, max: {param.max})")
        elif ("cheek" in param_id.lower() or "tongue" in param_id.lower() or "jaw" in param_id.lower()):
            special_params.append(param_id)
            print(f"Special param: {param_id} (min: {param.min}, max: {param.max})")

    if not mouth_params:
        mouth_params = ["ParamMouthOpenY", "PARAM_MOUTH_OPEN_Y"]
        print("No mouth parameters found, using defaults")

    model.SetExpression("exp_06")
    print("Recording with multiple audio support")
    print("Controls:")
    print("- L key: Start/restart lip sync with audio")
    print("- Arrow keys: Move model")
    print("- I/U keys: Zoom in/out")
    print("- R key: Reset pose")
    print("- E key: Reset expression")
    print("- Space key: Toggle auto animations")
    print("- Click: Change expression")
    print("- Drag: Move the model")
    print("- ESC: Stop recording and exit")
    
    # Prepare for recording
    frame_count = 0
    recording_active = True
    clock = pygame.time.Clock()
    start_time = time.time()
    
    wav_instance_count = 0  # To track multiple instances of the same audio

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                
                hit_parts = model.HitPart(x, y, False)
                if hit_parts:
                    current_top_clicked_part_id = hit_parts[0]
                
                if event.button == 1:
                    model.SetRandomExpression()
                    model.StartRandomMotion("TapBody", 3)
                    print("Changed expression")
                
            if event.type == pygame.MOUSEMOTION:
                model.Drag(*pygame.mouse.get_pos())
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    dx -= 0.1
                elif event.key == pygame.K_RIGHT:
                    dx += 0.1
                elif event.key == pygame.K_UP:
                    dy += 0.1
                elif event.key == pygame.K_DOWN:
                    dy -= 0.1
                elif event.key == pygame.K_i:
                    scale += 0.1
                elif event.key == pygame.K_u:
                    scale -= 0.1
                elif event.key == pygame.K_r:
                    model.StopAllMotions()
                    model.ResetPose()
                    print("Pose reset")
                elif event.key == pygame.K_e:
                    model.ResetExpression()
                    print("Expression reset")
                elif event.key == pygame.K_l:
                    # Calculate the time for this audio segment
                    current_time = frame_count / fps
                    
                    # Create a copy of the audio file for this instance
                    wav_instance_count += 1
                    audio_instance = os.path.join(audio_dir, f"audio_{wav_instance_count}.wav")
                    shutil.copy(audio_path, audio_instance)
                    
                    # Record this audio segment
                    audio_segments.append({
                        'file': audio_instance,
                        'start_time': current_time,
                        'start_frame': frame_count,
                        'duration': audio_duration
                    })
                    
                    # Start audio playback for user to hear
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    
                    # Start lip sync
                    wav_handler = WavHandler()  # Create a new instance to start fresh
                    wav_handler.Start(audio_path)
                    lip_sync_active = True
                    
                    print(f"Started lip sync with audio at frame {frame_count} (time: {current_time:.2f}s)")
                    
                elif event.key == pygame.K_SPACE:
                    blink = not model.GetAutoBlinkEnable()
                    breath = not model.GetAutoBreathEnable()
                    model.SetAutoBlinkEnable(blink)
                    model.SetAutoBreathEnable(breath)
                    print(f"Auto animations: {'on' if blink else 'off'}")

        model.Update()

        if lip_sync_active and pygame.mixer.music.get_busy() and wav_handler.Update():
            rms = wav_handler.GetRms()

            for param_id in mouth_params:
                try:
                    if "openy" in param_id.lower():
                        model.AddParameterValue(param_id, rms * lip_sync_multiplier)
                    elif "form" in param_id.lower():
                        model.AddParameterValue(param_id, rms * 0.5)
                except Exception as e:
                    pass

            if vowel_params:
                try:
                    if "ParamA" in vowel_params:
                        model.AddParameterValue("ParamA", rms * 3.0 if rms > 0.05 else 0)
                    if "ParamO" in vowel_params:
                        model.AddParameterValue("ParamO", rms * 2.0 if 0.04 < rms < 0.15 else 0)
                    if "ParamI" in vowel_params:
                        model.AddParameterValue("ParamI", rms * 1.0 if rms < 0.06 else 0)
                    if "ParamU" in vowel_params:
                        model.AddParameterValue("ParamU", rms * 1.5 if 0.03 < rms < 0.1 else 0)
                    if "ParamE" in vowel_params:
                        model.AddParameterValue("ParamE", rms * 1.0 if 0.03 < rms < 0.08 else 0)
                except Exception as e:
                    pass

        if current_top_clicked_part_id is not None:
            try:
                idx = part_ids.index(current_top_clicked_part_id)
                model.SetPartOpacity(idx, 0.7)
                model.SetPartMultiplyColor(idx, 0.0, 0.0, 1.0, 0.9)
            except:
                pass

        model.SetOffset(dx, dy)
        model.SetScale(scale)
        live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
        model.Draw()

        # Always capture frames for the recording
        if recording_active:
            frame = capture_frame()
            frame_path = os.path.join(frames_dir, f'frame_{frame_count:05d}.png')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            if frame_count % fps == 0:
                elapsed = time.time() - start_time
                print(f"Recording: {frame_count/fps:.1f}s (real time: {elapsed:.1f}s)")
                
                # Also show active audio segments
                active_segments = 0
                current_time = frame_count / fps
                for seg in audio_segments:
                    if seg['start_time'] <= current_time < seg['start_time'] + seg['duration']:
                        active_segments += 1
                if active_segments > 0:
                    print(f"  Currently playing {active_segments} audio segments")

        pygame.display.flip()
        clock.tick(fps)

    print(f"Captured {frame_count} frames ({frame_count/fps:.1f} seconds)")
    
    if frame_count > 0 and audio_segments:
        # Save audio configuration for debugging
        with open(audio_config_file, 'w') as f:
            json.dump(audio_segments, f, indent=2)
        
        # Create the final video with multiple audio segments
        print("Creating final video with multiple audio segments...")
        create_video_with_multiple_audio(frames_dir, audio_segments, output_filename, fps)
    else:
        print("No frames or audio recorded. Did you press L during the session?")
    
    # Clean up temp files
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary files")
    except Exception as e:
        print(f"Warning: Could not clean up temp directory: {e}")
    
    live2d.dispose()
    pygame.quit()

if __name__ == "__main__":
    main()