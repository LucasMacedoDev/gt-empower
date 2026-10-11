import os, json, uuid, shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text, select
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from . import recognition as recog

STORAGE=Path(os.getenv('STORAGE_DIR','./storage')).resolve()
STORAGE.mkdir(parents=True,exist_ok=True)
DATABASE_URL=os.getenv('DATABASE_URL','sqlite:///./storage/gt_empower.db')
engine=create_engine(DATABASE_URL,connect_args={'check_same_thread':False} if DATABASE_URL.startswith('sqlite') else {},pool_pre_ping=True)
Session=sessionmaker(bind=engine)
Base=declarative_base()

class Person(Base):
    __tablename__='people'
    id=Column(String,primary_key=True)
    name=Column(String(160),nullable=False)
    category=Column(String(16),nullable=False)
    photo=Column(String(255),nullable=False)
    embedding=Column(Text,nullable=False)
    created_at=Column(DateTime,default=lambda:datetime.now(timezone.utc))

class Meeting(Base):
    __tablename__='meetings'
    id=Column(String,primary_key=True)
    date=Column(String(10),nullable=False)
    created_at=Column(DateTime,default=lambda:datetime.now(timezone.utc))

class Photo(Base):
    __tablename__='photos'
    id=Column(String,primary_key=True)
    meeting_id=Column(String,ForeignKey('meetings.id'),nullable=False)
    path=Column(String,nullable=False)

class Sighting(Base):
    __tablename__='sightings'
    id=Column(String,primary_key=True)
    meeting_id=Column(String,ForeignKey('meetings.id'),nullable=False)
    photo_id=Column(String,ForeignKey('photos.id'),nullable=False)
    face_index=Column(Integer,nullable=False)
    box=Column(Text,nullable=False)
    embedding=Column(Text,nullable=False)
    person_id=Column(String,ForeignKey('people.id'),nullable=True)
    status=Column(String(16),nullable=False,default='unknown')
    distance=Column(String,nullable=True)

Base.metadata.create_all(engine)
app=FastAPI(title='GT Empower API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173').split(','),allow_methods=['*'],allow_headers=['*'])

def fail(e):
    if isinstance(e, HTTPException): raise e
    # Inform user of unavailable model / download / missing runtime instead of fake matches.
    raise HTTPException(422,detail=f'Falha no processamento facial real: {e}')

async def save_file(file:UploadFile):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415,'Envie JPEG, PNG ou WebP.')
    data=await file.read(15_000_001)
    if len(data)>15_000_000: raise HTTPException(413,'Arquivo maior que 15 MB.')
    name=f'{uuid.uuid4().hex}{ {"image/jpeg":".jpg","image/png":".png","image/webp":".webp"}[file.content_type]}'
    return name,data

def serialize_person(p):
    return {'id':p.id,'name':p.name,'category':p.category,'photo_url':'/api/files/'+p.photo,'created_at':p.created_at.isoformat() if p.created_at else None}

def best_match(embedding,people):
    ranked=sorted(((recog.distance(embedding,json.loads(p.embedding)),p) for p in people),key=lambda x:x[0])
    return ranked[0] if ranked else (float('inf'),None)

def classify(embedding,people):
    dist, p=best_match(embedding,people)
    if p and dist<=recog.THRESHOLD: return p,'identified',dist
    return None,'unknown',dist if p else None

@app.get('/api/health')
def health():
    try:
        recog.deepface()
        available=True; error=None
    except Exception as e:
        available=False; error=str(e)
    return {'api':'ok','deepface_importable':available,'deepface_error':error,'model':recog.MODEL,'detector':recog.DETECTOR,'cosine_threshold':recog.THRESHOLD,
            'notice':'Primeira análise pode baixar pesos de modelos da internet.'}

@app.get('/api/people')
def get_people():
    with Session() as db: return [serialize_person(p) for p in db.scalars(select(Person).order_by(Person.name)).all()]

@app.post('/api/people/inspect')
async def inspect_face(photo:UploadFile=File(...)):
    _,data=await save_file(photo)
    try: results=recog.faces(data)
    except Exception as e: fail(e)
    if len(results)!=1: raise HTTPException(422,f'Para cadastro, envie uma foto com exatamente um rosto. Encontrados: {len(results)}.')
    with Session() as db:
        d,p=best_match(results[0]['embedding'],db.scalars(select(Person)).all())
        duplicate=p is not None and d<=recog.THRESHOLD
        return {'detected_faces':1,'box':results[0]['box'],'duplicate':duplicate,
                'matching_person':serialize_person(p) if duplicate else None,'distance':d if p else None,
                'threshold':recog.THRESHOLD}

@app.post('/api/people')
async def create_person(name:str=Form(...),category:Literal['Membro','Visitante','Criança']=Form(...),photo:UploadFile=File(...)):
    name=name.strip()
    if not (2<=len(name)<=160): raise HTTPException(422,'Nome deve ter de 2 a 160 caracteres.')
    filename,data=await save_file(photo)
    try: detected=recog.faces(data)
    except Exception as e: fail(e)
    if len(detected)!=1: raise HTTPException(422,f'Cadastro exige exatamente um rosto (detectados: {len(detected)}).')
    emb=detected[0]['embedding']
    with Session() as db:
        dist,existing=best_match(emb,db.scalars(select(Person)).all())
        if existing and dist<=recog.THRESHOLD:
            raise HTTPException(409,detail={'message':'Rosto já cadastrado.','person':serialize_person(existing),'distance':dist})
        (STORAGE/filename).write_bytes(data)
        p=Person(id=uuid.uuid4().hex,name=name,category=category,photo=filename,embedding=json.dumps(emb))
        db.add(p);db.commit();return serialize_person(p)

@app.post('/api/people/verify')
async def verify_face(person_id:str=Form(...),photo:UploadFile=File(...)):
    _,data=await save_file(photo)
    try: found=recog.faces(data)
    except Exception as e: fail(e)
    if len(found)!=1: raise HTTPException(422,f'O teste exige exatamente um rosto (detectados: {len(found)}).')
    with Session() as db:
        person=db.get(Person,person_id)
        if not person: raise HTTPException(404,'Pessoa não encontrada.')
        d=recog.distance(found[0]['embedding'],json.loads(person.embedding))
        return {'matched':d<=recog.THRESHOLD,'distance':d,'threshold':recog.THRESHOLD,'person':serialize_person(person)}

@app.get('/api/meetings')
def meetings():
    with Session() as db:
        out=[]
        for m in db.scalars(select(Meeting).order_by(Meeting.date.desc(),Meeting.created_at.desc())).all():
            sightings=db.scalars(select(Sighting).where(Sighting.meeting_id==m.id)).all()
            out.append({'id':m.id,'date':m.date,'photos':len(db.scalars(select(Photo).where(Photo.meeting_id==m.id)).all()),
                        'unreviewed':sum(s.status=='unknown' for s in sightings)})
        return out

@app.post('/api/meetings')
def new_meeting(payload:dict):
    date=payload.get('date','')
    try: datetime.strptime(date,'%Y-%m-%d')
    except Exception: raise HTTPException(422,'Data inválida; use AAAA-MM-DD.')
    with Session() as db:
        m=Meeting(id=uuid.uuid4().hex,date=date);db.add(m);db.commit();return {'id':m.id,'date':date}

@app.post('/api/meetings/{meeting_id}/photos')
async def upload_meeting_photo(meeting_id:str,photo:UploadFile=File(...)):
    filename,data=await save_file(photo)
    with Session() as db:
        if not db.get(Meeting,meeting_id): raise HTTPException(404,'Reunião não encontrada.')
        try: found=recog.faces(data)
        except Exception as e: fail(e)
        (STORAGE/filename).write_bytes(data)
        ph=Photo(id=uuid.uuid4().hex,meeting_id=meeting_id,path=filename)
        db.add(ph)
        people=db.scalars(select(Person)).all()
        for i,f in enumerate(found):
            person,status,d=classify(f['embedding'],people)
            db.add(Sighting(id=uuid.uuid4().hex,meeting_id=meeting_id,photo_id=ph.id,
                face_index=i,box=json.dumps(f['box']),embedding=json.dumps(f['embedding']),
                person_id=person.id if person else None,status=status,distance=str(d) if d is not None else None))
        db.commit()
        return {'photo_id':ph.id,'faces_detected':len(found)}

@app.get('/api/meetings/{meeting_id}')
def get_meeting(meeting_id:str):
    with Session() as db:
        m=db.get(Meeting,meeting_id)
        if not m: raise HTTPException(404,'Reunião não encontrada.')
        photos=db.scalars(select(Photo).where(Photo.meeting_id==meeting_id)).all()
        s=db.scalars(select(Sighting).where(Sighting.meeting_id==meeting_id)).all()
        people={p.id:p for p in db.scalars(select(Person)).all()}
        return {'id':m.id,'date':m.date,'photos':[{'id':ph.id,'url':'/api/files/'+ph.path} for ph in photos],
            'sightings':[{'id':x.id,'photo_id':x.photo_id,'face_index':x.face_index,'box':json.loads(x.box),
            'status':x.status,'person_id':x.person_id,'person':serialize_person(people[x.person_id]) if x.person_id in people else None,
            'distance':float(x.distance) if x.distance else None} for x in s]}

class Resolve(BaseModel):
    person_id:str|None=None
    ignore:bool=False

@app.patch('/api/sightings/{sighting_id}')
def resolve(sighting_id:str,payload:Resolve):
    with Session() as db:
        s=db.get(Sighting,sighting_id)
        if not s: raise HTTPException(404,'Detecção não encontrada.')
        if payload.ignore:
            s.status='ignored';s.person_id=None;s.distance=None
        else:
            if not payload.person_id: raise HTTPException(422,'Escolha pessoa ou marque ignorar.')
            person=db.get(Person,payload.person_id)
            if not person: raise HTTPException(404,'Pessoa não encontrada.')
            d=recog.distance(json.loads(s.embedding),json.loads(person.embedding))
            s.status='reviewed';s.person_id=person.id;s.distance=str(d)
        db.commit()
        return {'ok':True}

@app.post('/api/sightings/{sighting_id}/enroll')
async def enroll_sighting(sighting_id:str,name:str=Form(...),category:Literal['Membro','Visitante','Criança']=Form(...)):
    name=name.strip()
    if len(name)<2 or len(name)>160: raise HTTPException(422,'Nome inválido.')
    with Session() as db:
        s=db.get(Sighting,sighting_id)
        if not s: raise HTTPException(404,'Detecção não encontrada.')
        emb=json.loads(s.embedding)
        dist,p=best_match(emb,db.scalars(select(Person)).all())
        if p and dist<=recog.THRESHOLD:
            raise HTTPException(409,detail={'message':'Rosto existente; vincule à pessoa em vez de cadastrar novamente.','person':serialize_person(p)})
        ph=db.get(Photo,s.photo_id)
        # Crop original face image strictly to this detection; enrollment retains a real photo.
        from PIL import Image,ImageOps
        import io
        original=ImageOps.exif_transpose(Image.open(STORAGE/ph.path)).convert('RGB')
        box=json.loads(s.box); x,y,w,h=box['x'],box['y'],box['w'],box['h']
        crop=original.crop((max(0,x),max(0,y),min(original.width,x+w),min(original.height,y+h)))
        filename=f'{uuid.uuid4().hex}.jpg';crop.save(STORAGE/filename,quality=93)
        person=Person(id=uuid.uuid4().hex,name=name,category=category,photo=filename,embedding=s.embedding)
        db.add(person)
        # Link all sightings in this meeting that are genuinely within threshold to newly registered face.
        for other in db.scalars(select(Sighting).where(Sighting.meeting_id==s.meeting_id,Sighting.status=='unknown')).all():
            d=recog.distance(json.loads(other.embedding),emb)
            if d<=recog.THRESHOLD:
                other.person_id=person.id;other.status='reviewed';other.distance=str(d)
        db.commit();return serialize_person(person)

@app.get('/api/files/{filename}')
def media(filename:str):
    if '/' in filename or '\\' in filename or filename.startswith('.'):
        raise HTTPException(400,'Arquivo inválido.')
    path=STORAGE/filename
    if not path.is_file():raise HTTPException(404,'Arquivo não encontrado.')
    return FileResponse(path)
