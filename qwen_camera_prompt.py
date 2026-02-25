"""
Qwen Camera Prompt node for yaple-nodes.

Generates a single camera-angle prompt. Supports two LoRA formats toggled in the UI:

  fal   — fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
            Format: <sks> {azimuth} {elevation} {distance}

  dx8152 — dx8152/Qwen-Edit-2509-Multiple-angles
            Format: bilingual natural language, no trigger word
            (also used by the linoyts/Qwen-Image-Edit-Angles HuggingFace space)

The 3D viewport in the node UI exposes three draggable colored handles:
  Green  (●) — drag L/R  → azimuth   (8 positions, 0°–315° in 45° steps)
  Pink   (●) — drag U/D  → elevation  (4 positions: −30°, 0°, 30°, 60°)
  Orange (●) — drag U/D  → distance   (4 positions: close-up, forward, medium/neutral, wide)

Format is toggled via the [fal / 2511] [dx8152 / 2509] buttons in the node UI.
The active format is stored in the camera_state JSON as the "fmt" key.
"""

import json

# ── fal LoRA tables ────────────────────────────────────────────────────────────

AZIMUTHS = [
    "front view",               # 0°
    "front-right quarter view", # 45°
    "right side view",          # 90°
    "back-right quarter view",  # 135°
    "back view",                # 180°
    "back-left quarter view",   # 225°
    "left side view",           # 270°
    "front-left quarter view",  # 315°
]

ELEVATIONS = [
    "low-angle shot",   # −30°  camera below, looking up
    "eye-level shot",   #   0°  camera at subject level
    "elevated shot",    #  30°  camera above, looking down
    "high-angle shot",  #  60°  camera high above, looking down
]

DISTANCES = [
    "close-up",     # dist=0  ×0.6
    "medium shot",  # dist=1  ×1.0  (fal; dx8152: forward)
    "medium shot",  # dist=2  ×1.0  (fal; dx8152: neutral — default)
    "wide shot",    # dist=3  ×1.8
]

# ── dx8152 LoRA prompt maps ────────────────────────────────────────────────────
# Bilingual (Chinese + English) natural-language phrases used by the LoRA and
# by the linoyts/Qwen-Image-Edit-Angles HuggingFace space.
#
# Azimuth → relative rotation command.
# The dx8152 LoRA operates in ±90° relative space; azimuths past ±90°
# (135°, 180°, 225°) are clamped to the nearest ±90° equivalent.

_DX_AZ = [
    None,                                                                              # 0°   front — no rotation
    ("将镜头向右旋转45度", "Rotate the camera 45 degrees to the right."),             # 45°  front-right
    ("将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."),             # 90°  right side
    ("将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."),             # 135° back-right (clamped)
    ("将镜头向右旋转90度", "Rotate the camera 90 degrees to the right."),             # 180° back (clamped)
    ("将镜头向左旋转90度", "Rotate the camera 90 degrees to the left."),              # 225° back-left (clamped)
    ("将镜头向左旋转90度", "Rotate the camera 90 degrees to the left."),              # 270° left side
    ("将镜头向左旋转45度", "Rotate the camera 45 degrees to the left."),              # 315° front-left
]

# Elevation → tilt/view command.
_DX_EL = [
    ("将相机切换到仰视视角", "Turn the camera to a worm's-eye view."),    # −30° low-angle
    None,                                                                    #   0° eye-level — no command
    ("将镜头向上移动",       "Move the camera up."),                        #  30° elevated
    ("将相机转向鸟瞰视角",   "Turn the camera to a bird's-eye view."),      #  60° high-angle
]

# Distance → zoom/movement command.
# dist=2 (medium/neutral) is the default — no distance command.
# dist=1 moves forward (closer); dist=0 zooms in to close-up.
# dist=3 moves backward (wider than source), which the fal LoRA cannot do
# (fal uses absolute positions, so "wide shot" on a wide-shot source = no change).
_DX_DIST = [
    ("将镜头转为特写镜头", "Turn the camera to a close-up."),  # dist=0  close-up
    ("将镜头向前移动",     "Move the camera forward."),         # dist=1  forward
    None,                                                        # dist=2  medium — neutral, no command
    ("将镜头向后移动",     "Move the camera backward."),        # dist=3  wide — zoom out / go wider
]

_DX_NEUTRAL = "保持当前视角 Keep the current camera angle."


def _prompt_fal(az, el, dist):
    return f"<sks> {AZIMUTHS[az]} {ELEVATIONS[el]} {DISTANCES[dist]}"


def _prompt_dx8152(az, el, dist):
    parts = []
    rot  = _DX_AZ[az];   parts.append(f"{rot[0]} {rot[1]}")   if rot   else None
    tilt = _DX_EL[el];   parts.append(f"{tilt[0]} {tilt[1]}") if tilt  else None
    zoom = _DX_DIST[dist]; parts.append(f"{zoom[0]} {zoom[1]}") if zoom else None
    return " ".join(parts) if parts else _DX_NEUTRAL


class QwenCameraPrompt:
    """
    3D camera angle selector for Qwen Image Edit multi-angle generation.

    Supports two LoRA formats (toggle in the node UI):
      fal    → fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
      dx8152 → dx8152/Qwen-Edit-2509-Multiple-angles

    The hidden `camera_state` widget is written by the JS 3D viewport and
    encodes az, el, dist indices plus the active format as JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_state": ("STRING", {
                    "default": '{"az":0,"el":1,"dist":2,"fmt":"fal"}',
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "yaple/camera"

    def build_prompt(self, camera_state='{"az":0,"el":1,"dist":2,"fmt":"fal"}'):
        try:
            state = json.loads(camera_state)
        except (json.JSONDecodeError, TypeError, ValueError):
            state = {}

        az   = max(0, min(7, int(state.get("az",   0))))
        el   = max(0, min(3, int(state.get("el",   1))))
        dist = max(0, min(3, int(state.get("dist", 2))))
        fmt  = state.get("fmt", "fal")

        if fmt == "dx8152":
            prompt = _prompt_dx8152(az, el, dist)
        else:
            prompt = _prompt_fal(az, el, dist)

        print(f"[QwenCameraPrompt] [{fmt}] {prompt}")
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "QwenCameraPrompt": QwenCameraPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenCameraPrompt": "Qwen Camera Prompt",
}
