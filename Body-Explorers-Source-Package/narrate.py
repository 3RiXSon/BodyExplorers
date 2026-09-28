import asyncio,json,os,subprocess
from pathlib import Path
import edge_tts,edge_tts.communicate
P=Path(__file__).parent
if os.environ.get('SSL_CERT_FILE'):
 edge_tts.communicate._SSL_CTX.load_verify_locations(os.environ['SSL_CERT_FILE'])
S=json.loads((P/'script.json').read_text())
async def one(i,s,sem):
 async with sem:
  f=P/f'narration-{i:02}.mp3';bound=[]
  com=edge_tts.Communicate(s['text'],'en-US-AriaNeural',rate='+5%',boundary='WordBoundary')
  with f.open('wb') as out:
   async for chunk in com.stream():
    if chunk['type']=='audio':out.write(chunk['data'])
    elif chunk['type']=='WordBoundary':bound.append({k:chunk[k] for k in ['offset','duration','text']})
  d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(f)]));s['audio_duration']=d;s['words']=bound
  print(i,round(d,2),len(bound),flush=True)
async def main():
 sem=asyncio.Semaphore(3);await asyncio.gather(*(one(i,s,sem) for i,s in enumerate(S)))
 (P/'script.json').write_text(json.dumps(S,indent=2))
asyncio.run(main())
