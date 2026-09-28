from pathlib import Path
import subprocess,json,numpy as np,wave,math
P=Path(__file__).parent;S=json.loads((P/'script.json').read_text());SR=44100
# Fit each spoken scene inside its 15-second window without clipping.
voice=np.zeros(SR*300,dtype=np.float32)
for i,shot in enumerate(S):
 speed=max(1,shot['audio_duration']/14.0)
 pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(P/f'narration-{i:02}.mp3'),'-af',f'atempo={speed},asetpts=PTS-STARTPTS','-ar',str(SR),'-ac','1','-f','f32le','-'])
 samples=np.frombuffer(pcm,dtype='<f4');assert len(samples)/SR<14.4
 start=int((i*15+.5)*SR);voice[start:start+len(samples)]=samples
# Normalize the assembled speech once; retain an exact sample timeline.
rms=np.sqrt(np.mean(voice[voice!=0]**2));voice*=min(0.14/max(rms,1e-6),.86/max(np.max(np.abs(voice)),1e-6))
with wave.open(str(P/'voice.wav'),'wb') as out:
 out.setnchannels(1);out.setsampwidth(2);out.setframerate(SR);out.writeframes((voice*32767).astype('<i2').tobytes())
# Original, quiet pentatonic music; no sampled or third-party song.
length=300;music=np.zeros(SR*length,dtype=np.float32)
def note(at,midi,duration,vol):
 start=int(at*SR);n=min(int(duration*SR),len(music)-start)
 if n<=0:return
 t=np.arange(n)/SR;f=440*2**((midi-69)/12)
 wave=(np.sin(2*np.pi*f*t)+.23*np.sin(2*np.pi*f*2*t)+.08*np.sin(2*np.pi*f*3*t))*np.exp(-t*3.5)*np.minimum(t/.012,1)
 music[start:start+n]+=wave.astype('f4')*vol
chords=[[48,52,55,59],[45,48,52,55],[41,45,48,52],[43,47,50,55]]
beat=60/96
for bar in range(120):
 at=bar*4*beat
 if at>=length:break
 chord=chords[bar%4]
 for k,m in enumerate(chord):note(at+k*.12,m+12,2.5,.018)
 for k,m in enumerate([72,76,79,76] if bar%2==0 else [74,79,81,79]):note(at+k*beat,m,1,.009)
 note(at,chord[0]-12,2,.016)
for i in [8,16]:
 for j in np.arange(i*15,(i+1)*15,.8):note(j,32,.18,.019);note(j+.19,29,.18,.012)
fade=np.minimum(np.arange(len(music))/SR/2,1)*np.minimum((len(music)-np.arange(len(music)))/SR/3,1);music*=fade
with wave.open(str(P/'music.wav'),'wb') as out:out.setnchannels(1);out.setsampwidth(2);out.setframerate(SR);out.writeframes((np.clip(music,-1,1)*32767).astype('<i2').tobytes())
subprocess.run(['ffmpeg','-y','-v','error','-i',str(P/'silent.mp4'),'-i',str(P/'voice.wav'),'-i',str(P/'music.wav'),'-filter_complex','[1:a]asetpts=PTS-STARTPTS[voc];[2:a]asetpts=PTS-STARTPTS[mus];[voc][mus]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.94:level=false,apad,atrim=duration=300[a]','-map','0:v','-map','[a]','-c:v','copy','-c:a','aac','-b:a','160k','-t','300','-movflags','+faststart',str(P/'Body-Explorers-E01-The-Great-Oxygen-Delivery.mp4')],check=True)
print('Finished five-minute film',flush=True)
