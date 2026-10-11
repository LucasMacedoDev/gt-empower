"""Real DeepFace extraction/embedding. Never return simulated recognition results."""
import io
import os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageOps

MODEL = os.getenv('FACE_MODEL', 'Facenet512')
DETECTOR = os.getenv('FACE_DETECTOR', 'retinaface')
# DeepFace's model-specific cosine verification threshold; tune with local validation data.
THRESHOLD = float(os.getenv('FACE_COSINE_THRESHOLD', '0.30'))

@lru_cache(maxsize=1)
def deepface():
    # Lazy import: service can start and report capability errors even when ML is absent.
    from deepface import DeepFace
    return DeepFace

def _rgb(data: bytes):
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert('RGB')
    if im.width * im.height > 24_000_000:
        raise ValueError('Imagem muito grande (máximo de 24 megapixels).')
    return np.asarray(im)

def faces(data: bytes):
    """Return list of embeddings and bounding boxes for faces actually detected."""
    image = _rgb(data)
    found = deepface().extract_faces(img_path=image, detector_backend=DETECTOR,
        enforce_detection=True, align=True, color_face='rgb', normalize_face=False)
    out=[]
    for face in found:
        region=face['facial_area']
        # Use aligned crop; passing detector=skip here avoids detecting twice.
        crop=np.asarray(face['face'])
        if crop.dtype != np.uint8:
            crop=np.clip(crop * 255 if crop.max() <= 1 else crop, 0, 255).astype('uint8')
        emb=deepface().represent(img_path=crop, model_name=MODEL,
            detector_backend='skip', enforce_detection=False, align=False)[0]['embedding']
        out.append({'embedding':[float(v) for v in emb],
            'box':{'x':int(region['x']), 'y':int(region['y']),
                   'w':int(region['w']), 'h':int(region['h'])},
            'confidence':float(face.get('confidence',0))})
    return out

def distance(a,b):
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    if denominator == 0 or len(x) != len(y):
        return 1.0
    return float(1 - np.dot(x,y)/denominator)
