import os

import pygame
from pygame.locals import *

import live2d.v3 as live2d
from live2d.utils import log

from live2d.utils.lipsync import WavHandler


def main():
    pygame.init()
    pygame.mixer.init()
    live2d.init()

    # Set up display
    display = (1000, 1700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Live2D Viewer")

    if live2d.LIVE2D_VERSION == 3:
        live2d.glewInit()

    model = live2d.LAppModel()

    model.LoadModelJson(os.path.join("Mao/Mao.model3.json"))

    model.Resize(*display)

    running = True
    dx, dy = 0.0, 0.0
    scale = 1.0
    audio_path = os.path.join("audio1.wav")

    wav_handler = WavHandler()
    lip_sync_multiplier = 10.0  # Increase multiplier for more sensitivity

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

        elif (
            "cheek" in param_id.lower()
            or "tongue" in param_id.lower()
            or "jaw" in param_id.lower()
        ):
            special_params.append(param_id)
            print(f"Special param: {param_id} (min: {param.min}, max: {param.max})")

    if not mouth_params:
        mouth_params = ["ParamMouthOpenY", "PARAM_MOUTH_OPEN_Y"]
        print("No mouth parameters found, using defaults")

    clock = pygame.time.Clock()

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
                    model.SetExpression("exp_06")
                    model.StartRandomMotion("TapBody", 3)

                if event.button == 3:
                    model.SetExpression("exp_06")
                    model.StartRandomMotion("TapBody", 3)
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    wav_handler.Start(audio_path)

            if event.type == pygame.MOUSEMOTION:
                model.Drag(*pygame.mouse.get_pos())

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
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
                elif event.key == pygame.K_e:
                    model.ResetExpression()

                elif event.key == pygame.K_l:
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    wav_handler.Start(audio_path)

                elif event.key == pygame.K_SPACE:
                    blink = not model.GetAutoBlinkEnable()
                    breath = not model.GetAutoBreathEnable()
                    model.SetAutoBlinkEnable(blink)
                    model.SetAutoBreathEnable(breath)

        model.Update()

        if pygame.mixer.music.get_busy() and wav_handler.Update():
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
                        model.AddParameterValue(
                            "ParamA", rms * 3.0 if rms > 0.05 else 0
                        )
                    if "ParamO" in vowel_params:
                        model.AddParameterValue(
                            "ParamO", rms * 2.0 if 0.04 < rms < 0.15 else 0
                        )
                    if "ParamI" in vowel_params:
                        model.AddParameterValue(
                            "ParamI", rms * 1.0 if rms < 0.06 else 0
                        )
                    if "ParamU" in vowel_params:
                        model.AddParameterValue(
                            "ParamU", rms * 1.5 if 0.03 < rms < 0.1 else 0
                        )
                    if "ParamE" in vowel_params:
                        model.AddParameterValue(
                            "ParamE", rms * 1.0 if 0.03 < rms < 0.08 else 0
                        )
                except Exception as e:
                    pass

            print(f"RMS: {rms:.3f}")

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

        pygame.display.flip()
        clock.tick(60)

    live2d.dispose()
    pygame.quit()


if __name__ == "__main__":
    main()
