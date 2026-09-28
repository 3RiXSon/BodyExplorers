import os,math,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import moderngl
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent
W,H=1280,720; FPS=24; DUR=15
S=json.loads((P/'script.json').read_text())
ctx=moderngl.create_standalone_context(backend='egl')
fbo=ctx.simple_framebuffer((W,H),components=3); fbo.use();ctx.enable(moderngl.DEPTH_TEST)
prog=ctx.program(vertex_shader='''#version 330
in vec3 in_pos; in vec3 in_normal;
uniform mat4 model; uniform mat4 vp;
out vec3 N;out vec3 P;
void main(){vec4 p=model*vec4(in_pos,1);P=p.xyz;N=mat3(transpose(inverse(model)))*in_normal;gl_Position=vp*p;}
''',fragment_shader='''#version 330
in vec3 N;in vec3 P;uniform vec3 color;uniform vec3 eye;out vec4 frag;
void main(){vec3 n=normalize(N);vec3 l=normalize(vec3(-4,7,8)-P);vec3 v=normalize(eye-P);
float d=max(dot(n,l),0);float rim=pow(1-max(dot(n,v),0),3);
float spec=pow(max(dot(n,normalize(l+v)),0),45);
vec3 c=color*(.36+.62*d)+vec3(.25,.38,.43)*rim+vec3(.40)*spec;
frag=vec4(pow(c,vec3(.9)),1);}
''')
def mesh_sphere(n=22,m=32):
 verts=[];inds=[]
 for i in range(n+1):
  a=math.pi*i/n
  for j in range(m+1):
   b=2*math.pi*j/m;x=math.sin(a)*math.cos(b);y=math.cos(a);z=math.sin(a)*math.sin(b);verts.append([x,y,z,x,y,z])
 for i in range(n):
  for j in range(m):
   k=i*(m+1)+j;inds.extend([k,k+m+1,k+1,k+1,k+m+1,k+m+2])
 return np.array(verts,'f4'),np.array(inds,'i4')
def mesh_rbc():
 n,m=28,48;v=[];idx=[]
 for i in range(n+1):
  a=math.pi*i/n;r=math.sin(a);z=math.cos(a)*(.18+.58*r*r)
  for j in range(m+1):
   b=j*2*math.pi/m
   # finite difference surface normal for biconcave cell
   da=.0001;rr=math.sin(a+da);zz=math.cos(a+da)*(.18+.58*rr*rr)
   dr=(rr-r)/da;dz=(zz-z)/da
   normal=np.array([-dz*math.cos(b),-dz*math.sin(b),dr]);normal/=max(np.linalg.norm(normal),1e-6)
   v.append([r*math.cos(b),r*math.sin(b),z,*normal])
 for i in range(n):
  for j in range(m):
   k=i*(m+1)+j;idx.extend([k,k+m+1,k+1,k+1,k+m+1,k+m+2])
 return np.array(v,'f4'),np.array(idx,'i4')
def vao(data):
 v,i=data;return ctx.vertex_array(prog,[(ctx.buffer(v.tobytes()),'3f 3f','in_pos','in_normal')],ctx.buffer(i.tobytes()))
sphere=vao(mesh_sphere());rbcmesh=vao(mesh_rbc())
TEAL=(.07,.76,.76); GOLD=(1,.66,.16); RED=(.97,.16,.30); PINK=(.98,.40,.49); WHITE=(.96,.99,1); NAVY=(.035,.095,.17); PURPLE=(.54,.32,.79)
def draw(pos,scale,color,mesh=sphere,rot=0,mat=None):
 if mat is None:
  c,s=math.cos(rot),math.sin(rot);mat=np.array([[c*scale[0],-s*scale[1],0,pos[0]],[s*scale[0],c*scale[1],0,pos[1]],[0,0,scale[2],pos[2]],[0,0,0,1]],'f4')
 prog['model'].write(np.array(mat,'f4').T.tobytes());prog['color'].value=color;mesh.render()
def ball(p,r,c):draw(p,(r,r,r),c)
def tube(a,b,r,c):
 a=np.array(a);b=np.array(b);delta=b-a;d=np.linalg.norm(delta);up=delta/d;x=np.cross(up,[0,0,1])
 if np.linalg.norm(x)<.01:x=np.cross(up,[1,0,0])
 x/=np.linalg.norm(x);z=np.cross(x,up);mat=np.eye(4);mat[:3,0]=x*r;mat[:3,1]=up*(d/2+r);mat[:3,2]=z*r;mat[:3,3]=(a+b)/2;draw((0,0,0),(1,1,1),c,mat=mat)
def eyes(x,y,z,s,t):
 blink=.12 if (t%5.3)<.10 else 1
 for dx in [-.23,.23]:
  draw((x+dx*s,y+.12*s,z),(.145*s,.20*s*blink,.10*s),WHITE)
  draw((x+dx*s+.025*s*math.sin(t*.5),y+.10*s,z+.084*s),(.064*s,.105*s*blink,.045*s),NAVY)
 # smile made of a curved line
 for i in range(8):
  a=math.pi+math.pi*i/7; b=math.pi+math.pi*(i+1)/7
  tube((x+.19*s*math.cos(a),y-.17*s+.13*s*math.sin(a),z),(x+.19*s*math.cos(b),y-.17*s+.13*s*math.sin(b),z),.022*s,NAVY)
def ruby(x,y,z,s,t,loaded=False,stretch=1):
 bob=.10*math.sin(t*2);y+=bob
 draw((x,y,z),(s*stretch,s/stretch,s),RED,rbcmesh,rot=.1*math.sin(t))
 eyes(x,y,z+.42*s,s,t)
 for side in [-1,1]:
  a=(x+side*.78*s,y-.12*s,z);b=(x+side*1.25*s,y+(.05+.2*math.sin(t*2+side))*s,z+.12*s);tube(a,b,.072*s,RED);ball(b,.115*s,RED)
 if loaded:
  for k in range(5):
   a=k*2*math.pi/5+t*.35;ball((x+1.35*s*math.cos(a),y+1.35*s*math.sin(a),z-.1),.13*s,GOLD)
def pip(x,y,z,s,t):
 y+=.12*math.sin(t*1.8)
 draw((x,y-.65*s,z),(.52*s,.65*s,.38*s),TEAL)
 draw((x,y+.12*s,z),(.72*s,.67*s,.55*s),GOLD)
 draw((x,y+.14*s,z+.39*s),(.59*s,.48*s,.27*s),NAVY)
 for side in [-1,1]:
  draw((x+side*.23*s,y+.20*s,z+.65*s),(.092*s,.125*s,.035*s),(.45,1,.94))
  a=(x+side*.4*s,y-.55*s,z);b=(x+side*.86*s,y-.45*s+.2*s*math.sin(t*2+side),z+.12*s)
  tube(a,b,.12*s,TEAL);ball(b,.16*s,GOLD)
  tube((x+side*.21*s,y-1*s,z),(x+side*.3*s,y-1.38*s,z+.10),.14*s,TEAL)
  draw((x+side*.3*s,y-1.42*s,z+.18*s),(.23*s,.13*s,.29*s),GOLD)
 tube((x,y+.70*s,z),(x,y+.98*s,z),.04*s,GOLD);ball((x,y+1.03*s,z),.11*s,TEAL)
def heart(x,y,z,s,t):
 pulse=1+.07*max(0,math.sin(t*6))**4;s*=pulse
 for dx in [-.46,.46]:draw((x+dx*s,y+.36*s,z),(.66*s,.72*s,.55*s),RED)
 draw((x,y-.30*s,z),(.76*s,.88*s,.55*s),RED,rot=-.12)
 for dx,c in [(-.55,(.58,.16,.30)),(.55,PINK)]:
  tube((x+dx*s,y+.65*s,z-.1),(x+dx*s,y+1.40*s,z-.1),.19*s,c)
 tube((x+.26*s,y+.7*s,z),(x+.6*s,y+1.33*s,z),.16*s,GOLD)
 # surface coronary branch
 tube((x,y+.65*s,z+.53*s),(x-.15*s,y-.12*s,z+.57*s),.045*s,GOLD)
 tube((x-.15*s,y-.12*s,z+.57*s),(x+.32*s,y-.54*s,z+.46*s),.035*s,GOLD)
def lungs(x,y,z,s,t):
 b=1+.05*math.sin(t*1.7)
 for side in [-1,1]:
  draw((x+side*.85*s,y,z),(.70*s*b,1.28*s*b,.56*s),PINK,rot=-side*.10)
 tube((x,y+1.9*s,z+.1),(x,y+.60*s,z+.1),.16*s,(.90,.71,.59))
 for side in [-1,1]:
  tube((x,y+.66*s,z+.30),(x+side*.65*s,y+.13*s,z+.53),.10*s,(1,.85,.61))
  for k in range(3):
   tube((x+side*.5*s,y+.2*s,z+.55),(x+side*(.7+.18*k)*s,y+(.6-.55*k)*s,z+.55),.057*s,(1,.85,.61))
def alveoli(x,y,z,s,t):
 tube((x,y+2*s,z),(x,y+.5*s,z),.19*s,(.93,.66,.56))
 for k in range(12):
  a=2*math.pi*k/12;r=.82 if k%2 else 1.15
  pos=(x+r*s*math.cos(a),y+r*s*math.sin(a),z+.25*math.sin(a*2))
  ball(pos,.53*s*(1+.035*math.sin(t*1.5)),(.97,.59,.47))
  if k%2==0:tube((x,y+.5*s,z),pos,.08*s,(1,.76,.58))
 for k in range(20):
  a=k/20*2*math.pi;b=(k+1)/20*2*math.pi
  tube((x+1.62*s*math.cos(a),y+1.50*s*math.sin(a),z+.2),(x+1.62*s*math.cos(b),y+1.50*s*math.sin(b),z+.2),.085*s,RED)
def vessel(t,n=12):
 # receding vessel rings, open tunnel, blood always red
 for j in range(9):
  z=-14+j*2.2+(t*.9%2.2)
  for k in range(n):
   a=k/n*2*math.pi;b=(k+1)/n*2*math.pi
   tube((5.5*math.cos(a),3.35*math.sin(a),z),(5.5*math.cos(b),3.35*math.sin(b),z),.10,(.36,.12,.22))
 for j in range(6):
  x=((j*2.79+t*.7)%12)-6;y=-1.2+((j*1.61)%3)
  draw((x,y,-4-j%3),( .43,.40,.43),(.66,.13,.26),rbcmesh,rot=t*.2+j)
def muscle(x,y,z,s,t):
 f=1+.07*math.sin(t*3)
 for j in range(7):
  draw((x,y+(j-3)*.3*s,z+.1*math.cos(j)),(2.1*s/f,.20*s*f,.23*s*f),(.85,.19+.025*j,.30))
  for k in range(5):ball((x+(-1.5+k*.73)*s/f,y+(j-3)*.3*s,z+.20),.06*s,(1,.59,.55))
def kid(t):
 x=1.6;y=.45;z=0
 ball((x,y+1.48,z),.60,(.66,.39,.24));draw((x,y+1.83,z-.06),(.63,.35,.55),(.15,.09,.075))
 eyes(x,y+1.45,z+.53,.62,t)
 draw((x,y+.3,z),(.60,.87,.34),TEAL)
 for side in [-1,1]:
  swing=.55*math.sin(t*6+side*math.pi/2)
  tube((x+side*.49,y+.7,z),(x+side*.87,y+.15,z+swing),.14,TEAL)
  ball((x+side*.87,y+.15,z+swing),.17,(.66,.39,.24))
  tube((x+side*.25,y-.4,z),(x+side*.36,y-1.18,z-swing),.19,NAVY)
  tube((x+side*.36,y-1.18,z-swing),(x+side*.44,y-1.92,z+swing),.16,NAVY)
  draw((x+side*.44,y-1.98,z+swing+.16),(.25,.17,.4),GOLD)
def camera(t):
 eye=np.array([.35*math.sin(t*.15),.25,13.8]);target=np.array([0,0,0]);f=(target-eye);f/=np.linalg.norm(f);s=np.cross(f,[0,1,0]);s/=np.linalg.norm(s);u=np.cross(s,f)
 view=np.eye(4);view[0,:3]=s;view[1,:3]=u;view[2,:3]=-f;view[:3,3]=-view[:3,:3]@eye
 a=W/H;q=1/math.tan(math.radians(41)/2);near=.1;far=80
 pr=np.array([[q/a,0,0,0],[0,q,0,0],[0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]])
 prog['vp'].write(np.array(pr@view,'f4').T.tobytes());prog['eye'].value=tuple(eye)
def scene(i,t):
 ctx.clear(.025,.065,.115,1);camera(t+i*3)
 # floating motes and a lit stage give the series a consistent world
 for k in range(22):
  a=k*2.399;x=7*math.cos(a);y=3*math.sin(a)+.12*math.sin(t+k);z=-3-(k%4)
  ball((x,y,z),.027+.02*(k%3),(.10,.29,.36))
 draw((0,-2.55,-1),(5.7,.18,2.1),(.04,.16,.23))
 typ=S[i]['scene'];p=t/DUR
 if typ in ['intro','outro']:
  pip(-2.4,.1,0,1.3,t);ruby(1.6,.3,0,1.6,t,True)
  for k in range(9):
   a=k*2*math.pi/9+t*.25;ball((4.6*math.cos(a),2.1*math.sin(a),-3),.12,GOLD if k%2 else TEAL)
 elif typ in ['ruby','scale','load']:
  pip(-3.6,-.1,0,1,t);ruby(1,.15,0,1.9 if typ!='scale' else 1.7,t,typ=='load')
  if typ=='scale':
   for k in range(5):draw((3.4+.55*math.sin(k+t),-1.8+k*.72,-2),(.25,.25,.25),RED,rbcmesh)
  if typ=='load':
   for k in range(5):
    a=(t*.4+k*.9)%3;ball((4.6-a,.4+math.sin(k*2)*1.5,.7),.14,GOLD)
 elif typ in ['lungs','exhale']:
  lungs(1.1,-.2,0,1.25,t);pip(-3.6,-.45,0,1.1,t)
  for k in range(10):
   f=(t*.3+k/10)%1;x=1.1+(.35*math.sin(k) if f<.45 else (1 if k%2 else -1)*(f-.45)*2)
   y=3.5-f*4.4 if typ=='lungs' else -.8+f*4.4
   ball((x,y,.85),.10,GOLD if typ=='lungs' else (.53,.73,.92))
 elif typ=='alveoli':
  alveoli(1.7,.0,0,1.22,t);pip(-3.4,-.5,0,1.2,t)
 elif typ in ['exchange','delivery']:
  pip(-4.4,-.65,0,.85,t)
  if typ=='exchange':alveoli(-1.7,.5,-1,.85,t);ruby(2.7,-.2,0,1.2,t,True)
  else:ruby(-1.8,.2,0,1.25,t,False);muscle(2.6,.1,0,.9,t)
  for k in range(6):
   f=(t*.22+k/6)%1;ball((-1.4+3.5*f,.4+.35*math.sin(f*math.pi*2+k),1),.12,GOLD)
 elif typ in ['travel','vessels','return']:
  vessel(t);ruby(.8,.1,1,1.45,t,typ!='return');pip(-2.55,-.25,1,1,t)
  if typ=='return':
   for k in range(8):ball((((k*1.5+t*.8)%11)-5.5,1.7*math.sin(k*2),.4),.12,(.52,.72,.88))
 elif typ in ['heart','rightheart']:
  heart(1.3,-.1,0,1.7,t);pip(-3.3,-.3,.4,1.1,t)
  if typ=='rightheart':ruby(-.8,-1.5,1,.5,t)
 elif typ=='circuit':
  heart(0,-.45,0,1.2,t);lungs(3.25,.2,-.5,.86,t);muscle(-3.3,.3,-.5,.65,t)
  route=[(3.25,.2,0),(0,-.6,0),(-3.3,.3,0),(0,-.6,0),(3.25,.2,0)]
  phase=(t/3)%4;idx=int(phase);f=phase-idx
  a=np.array(route[idx]);b=np.array(route[idx+1]);pos=a+(b-a)*f;pos[1]+=.85*math.sin(f*math.pi)*(1 if idx%2==0 else -1);pos[2]=1.0
  draw(pos,(.37,.37,.37),RED,rbcmesh)
 elif typ=='capillary':
  for side in [-1,1]:tube((-6,side*.96,-.5),(6,side*.96,-.5),.23,(.48,.14,.24))
  x=-2+4*p;ruby(x,0,0,1.05,t,True,1.35);pip(-3.8,1.4,0,.6,t)
 elif typ=='energy':
  draw((1.65,0,-.2),(2,1.65,.7),(.26,.49,.74))
  for k in range(5):
   a=k*2.4;draw((1.65+1.2*math.cos(a),1*math.sin(a),.5),(.55,.22,.19),GOLD,rot=a+t*.08)
  pip(-3.25,-.25,0,1.2,t)
  for k in range(8):a=k*2.4+t*.5;ball((1.65+1.75*math.cos(a),1.45*math.sin(a),.8),.09,TEAL)
 elif typ=='run':
  kid(t);pip(-3.3,-.5,0,1,t)
  for k in range(9):draw((((k*1.4-t*2)%14)-7,-2.28,.1),(.4,.03,.35),(.2,.43,.47))
 elif typ=='quiz':
  q=0 if t<5.85 else (1 if t<9.15 else 2)
  if q==0:lungs(-2.5,-.2,0,1.05,t);draw((2.6,0,0),(1,1.3,.6),PURPLE);tube((2.15,1.5,0),(2.15,.65,0),.18,PURPLE)
  elif q==1:
   heart(-2.5,-.2,0,1.3,t);tube((2.5,-1.4,0),(2.5,1.2,0),.35,WHITE)
   for side in [-1,1]:
    for yy in [-1.5,1.3]:ball((2.5+side*.3,yy,0),.42,WHITE)
  else:ruby(0,.15,0,1.75,t,True);pip(-3.4,-.5,0,1,t)
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
boldpath='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F=lambda n,b=False:ImageFont.truetype(boldpath if b else fontpath,n)
fonts={n:F(n) for n in [20,23,26,29,32]};titlefont=F(39,True)
def wrap(text,font,maxwidth):
 lines=[];line=''
 for word in text.split():
  test=(line+' '+word).strip()
  if font.getlength(test)>maxwidth:lines.append(line);line=word
  else:line=test
 if line:lines.append(line)
 return lines
# Readable captions in short clauses, timed within each 15-second shot.
CAP=[]
for shot in S:
 words=shot.get('words',[]);rate=max(1,shot['audio_duration']/14.0);arr=[]
 for k in range(0,len(words),9):
  group=words[k:k+9];a=.5+group[0]['offset']/1e7/rate
  b=.5+(group[-1]['offset']+group[-1]['duration'])/1e7/rate
  txt=' '.join(w['text'] for w in group)
  arr.append((a,min(14.9,b+.12),txt))
 CAP.append(arr)
cache={}
def overlay(i,t):
 ci=next((k for k,(a,b,c) in enumerate(CAP[i]) if a<=t<b),-1);qi=(0 if t<5.85 else (1 if t<9.15 else 2)) if S[i]['scene']=='quiz' else -1
 key=(i,ci,qi)
 if key in cache:return cache[key]
 im=Image.new('RGBA',(W,H));d=ImageDraw.Draw(im)
 d.rounded_rectangle((45,30,256,65),radius=16,fill=(18,62,76,240));d.text((61,37),'BODY EXPLORERS',font=F(19,True),fill=(104,238,220))
 d.text((W-146,39),f'EP 01  /  {i+1:02}',font=fonts[20],fill=(161,185,202))
 title=S[i]['title'];d.text((W/2,109),title,font=titlefont,anchor='mm',fill=(255,250,233))
 d.text((W/2,155),S[i]['tag'],font=fonts[23],anchor='mm',fill=(137,222,219))
 if ci>=0:
  txt=CAP[i][ci][2];lines=wrap(txt,fonts[29],1100)
  hh=42*len(lines)+24;top=H-42-hh
  d.rounded_rectangle((60,top,W-60,H-36),radius=18,fill=(3,15,27,231))
  for j,line in enumerate(lines):d.text((W/2,top+17+j*42),line,font=fonts[29],anchor='mt',fill=(255,255,246))
 if qi>=0:
  labels=[('LUNGS','STOMACH'),('HEART','BONE'),('RED BLOOD CELLS','')][qi]
  if qi<2:
   for x,label in zip([420,861],labels):d.text((x,526),label,font=F(26,True),anchor='mm',fill=(255,215,105))
 d.rounded_rectangle((46,H-18,W-46,H-12),radius=3,fill=(31,61,79))
 cache[key]=im;return im

def frame(i,t):
 scene(i,t);im=Image.frombytes('RGB',(W,H),fbo.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGBA');im.alpha_composite(overlay(i,t));d=ImageDraw.Draw(im);d.rounded_rectangle((46,H-18,46+(W-92)*(i+t/15)/20,H-12),radius=3,fill=(83,220,203));return im.convert('RGB')
def stamp(sec):
 ms=round(sec*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
if __name__=='__main__':
 if '--stills' in sys.argv:
  for i in [0,3,4,5,8,9,11,13,14,18]:frame(i,7).save(P/f'preview-{i:02}.jpg')
  print('stills ready')
 else:
  cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p',str(P/'silent.mp4')]
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.time()
  for i in range(20):
   for f in range(FPS*DUR):proc.stdin.write(frame(i,f/FPS).tobytes())
   print(f'Scene {i+1}/20 rendered in {time.time()-start:.1f}s',flush=True)
  proc.stdin.close();assert proc.wait()==0
  lines=[];idx=1
  for i,arr in enumerate(CAP):
   for a,b,c in arr:lines.append(f'{idx}\n{stamp(i*15+a)} --> {stamp(i*15+b)}\n{c}\n');idx+=1
  (P/'Body-Explorers-E01-Captions.srt').write_text('\n'.join(lines))
