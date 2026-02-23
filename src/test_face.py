import numpy as np
import cv2
import ssl
from insightface.app import FaceAnalysis


ssl._create_default_https_context = ssl._create_unverified_context



app = FaceAnalysis(name='buffalo_l',
    providers=['CPUExecutionProvider']
    )

app.prepare(ctx_id=0, det_size=(640, 640))


img=cv2.imread("data/me.jpeg")

if img is None:
    print("Error")

else:

    print("Image loaded successfully!")

    faces=app.get(img)

    if len(faces) ==0:
        print("No faces detected in the image.")
    else:
        print(f"Success, Detecef {len(faces)} face(s)")


