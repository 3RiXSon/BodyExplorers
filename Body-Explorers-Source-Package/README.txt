BODY EXPLORERS — EPISODE 1 SOURCE PROJECT

An original five-minute, 3D educational animation.
Output: 1280x720, 24 fps, H.264 video + AAC audio.

Requirements:
Python 3.12; ffmpeg/ffprobe; an OpenGL 3.3 EGL implementation.
Python packages: numpy, Pillow, moderngl, edge-tts.

To reproduce with the included narration files:
1. Run: python render.py
2. Run: python finish.py

render.py makes the silent animated video and matching captions.
finish.py fits narration into scenes, generates original instrumental
music, and combines the tracks into the final MP4.

To change the narration:
Edit script.json, then run python narrate.py before steps 1 and 2.
Narration generation needs internet access to the speech service.
The existing narration-00.mp3 through narration-19.mp3 are included,
so regenerating speech is not required for the supplied episode.

The models are procedural 3D meshes, rendered with perspective,
lighting, and animated transforms. They are simplified teaching models,
not anatomically exact or proportionally scaled medical illustrations.
No outside character art, stock footage, or sampled songs is used.
Narration is synthesized using the en-US-AriaNeural voice.

The episode guide contains the full narration, season plan, sources,
and proposed YouTube title, description, and chapter timestamps.
