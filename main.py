import os
import threading as td
import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *

import live2d.v3 as live2d
from facial_params import Params
from live2d.v3.params import StandardParams
from mediapipe_capture.capture_task import mediapipe_capture_task

live2d.setLogEnable(True)

width, height = 450, 700  
fps = 30
output_filename = "face_tracking_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

def capture_frame():
    """Captures OpenGL frame as an image buffer (RGB format)."""
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    return np.flipud(image)  

def draw():
    pygame.display.flip()
    pygame.time.wait(10)

def main():
    pygame.init()
    live2d.init()

    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    live2d.glewInit()
    model = live2d.LAppModel()
    model.LoadModelJson(os.path.join("Mao/Mao.model3.json"))
    model.Resize(*display)

    running = True
    params = None

    frame_count = 0
    max_frames = fps * 10  

    while running and frame_count < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                print("set random expression")
                model.SetRandomExpression()

        if not params:
            params = Params()
            td.Thread(None, mediapipe_capture_task, "Capture Task", (params,), daemon=True).start()

        model.Update()
        if params:
            params.update_params(params)
            model.SetParameterValue(StandardParams.ParamEyeLOpen, params.EyeLOpen, 1)
            model.SetParameterValue(StandardParams.ParamEyeROpen, params.EyeROpen, 1)
            model.SetParameterValue(StandardParams.ParamMouthOpenY, params.MouthOpenY, 1)
            model.SetParameterValue(StandardParams.ParamAngleX, params.AngleX, 1)
            model.SetParameterValue(StandardParams.ParamAngleY, params.AngleY, 1)
            model.SetParameterValue(StandardParams.ParamAngleZ, params.AngleZ, 1)
            model.SetParameterValue(StandardParams.ParamBodyAngleX, params.BodyAngleX, 1)

        model.SetParameterValue("Param14", 1, 1)  

        live2d.clearBuffer()
        model.Draw()
        draw()

        frame = capture_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        video_writer.write(frame_bgr)  

        frame_count += 1

    video_writer.release()
    print(f"Video saved as {output_filename}")
    live2d.dispose()
    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
