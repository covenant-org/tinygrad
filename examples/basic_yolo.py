#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
import cv2
import numpy as np
import time 
import onnx
import onnxruntime

base_path = Path(__file__).resolve().parent.parent / Path("models")
model_path = base_path / Path("yoloV8-Medium-NucleaV9/best.onnx")
# frame_path = "/home/manuelo247/models/ground-Truth/images/Avances-Elite-Toluquilla-II-a-julio-2024_mp4-0021.jpg"
gt_path = base_path / Path("ground-truth")

images_sample = sorted([img for ext in ["*.jpg", "*.jpeg", "*.png"] for img in gt_path.glob(ext)])
# model_onnx = onnx.load(model_path)
# for input in model_onnx.graph.input:
#     print(input.name, input.type.tensor_type.shape)

model = YOLO(str(model_path))

onnx_model = onnx.load(open(model_path, "rb"))
input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
target_size = tuple(reversed(input_shapes['images'][2:]))
print("input_shapes:", input_shapes)

session = onnxruntime.InferenceSession(str(model_path))
for i, inp in enumerate(session.get_inputs()):
    print(f"Input {i}: {inp.name}, shape={inp.shape}")

for frame_path in images_sample:
    image = cv2.imread(str(frame_path))
    print(frame_path)
    if image is None:
        print(f"Imagen \"{frame_path}\" no encontrada")
        continue
    resized_image = cv2.resize(image, target_size)
    resized_image = resized_image.transpose(2, 0, 1)  # Convertir a formato (C, H, W)
    resized_image = np.expand_dims(resized_image, axis=0)  # Añadir la dimensión de batch
    resized_image = resized_image.astype(np.float32)
    resized_image = resized_image.transpose(0, 2, 3, 1)  # Convertir a NHWC (1, 480, 640, 3)


    print("resized_image.shape:",resized_image.shape)
    st = time.time()
    results = model.predict(image, task="detect", verbose=True)
    print(f'Inferencia hecha en {int(round(((time.time() - st) * 1000)))}ms')
    # print("Resultado:",results)
    yolo_image = results[0].plot()
    resized_image = cv2.resize(yolo_image, target_size)
    # print("plot", yolo_image)
    cv2.imshow("ONNX model", resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()