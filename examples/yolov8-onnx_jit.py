#!/usr/bin/env python3
import os
import sys
import pickle
import time
import cv2
import onnx
import numpy as np
import yaml
import requests
from pathlib import Path
import zipfile
import shutil
import argparse
from onnx.helper import tensor_dtype_to_np_dtype

# Agregar directorio de 'extra'
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar módulos de 'extra' y 'tinygrad'
from extra.onnx import OnnxRunner
from tinygrad import Device, TinyJit, Context, GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG, getenv
from tinygrad.engine.realize import CompiledRunner

# Mostrar dispositivos disponibles
print(list(Device.get_available_devices()))

# Establecer variables de entorno si no están definidas
os.environ.setdefault("JIT", "2")
os.environ.setdefault("IMAGE", "1")
os.environ.setdefault("FLOAT16", "1")
os.environ.setdefault("DEFAULT_FLOAT", "HALF")
os.environ.setdefault("PREREALIZE", "0")

# os.environ.setdefault("CUDA", "1")
# os.environ.setdefault("DEBUG", "4")
# os.environ.setdefault("NOLOCALS", "1") # REDUCE OPTIMIZACION
# os.environ.setdefault("PTX", "1")
# os.environ.setdefault("PROFILE", "1")
# os.environ.setdefault("JIT_BATCH_SIZE", "0")

# Crear el parser
parser = argparse.ArgumentParser(description="Descargar modelo y dataset")
# Agregar argumentos
parser.add_argument("--url_model", type=str, default="https://github.com/covenant-org/tinygrad/releases/download/yoloV8-Medium-NucleaV9/best.onnx", help="URL del modelo ONNX")
parser.add_argument("--url_dataset", type=str, default="https://app.roboflow.com/ds/qnXOxt8VKv?key=1mmF2G81LD", help="URL del dataset")
parser.add_argument("--model_path_pkl", type=str, default="", help="Direccion del pkl")
parser.add_argument("--imshow", type=str, default="True", help="Mostrar imágenes a tiempo real")
parser.add_argument("--device", type=str, default="", help="Mostrar imágenes a tiempo real")
args = parser.parse_args()

args.imshow = args.imshow.lower() in ["true", "1", "yes"]
Device.DEFAULT = args.device

# Usar los valores en el código
print(f"URL del modelo: {args.url_model}")
print(f"URL del dataset: {args.url_dataset}")

debug = False

class Timer:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        """Inicia el timer."""
        self.start_time = time.time()
    
    def stop(self):
        """Detiene el timer y devuelve el tiempo transcurrido en segundos."""
        if self.start_time is None:
            raise ValueError("El timer no ha sido iniciado. Usa start() primero.")
        return int(round(((time.time() - self.start_time) * 1000)))

class ModelDownloader:
    def __init__(self, url_model: str = None, url_dataset: str = None):
        self.base_path = Path(__file__).resolve().parent.parent / "models"
        self.url_model = url_model
        if self.url_model:
            self.model_name = self.url_model.split("/")[-2]
        self.model_path_onnx = self.base_path / self.model_name / "best.onnx"
        self.zip_path = self.base_path / "dataset.zip"
        self.gt_path = self.base_path / "ground-truth"
        self.url_dataset = url_dataset
        
        self.base_path.mkdir(parents=True, exist_ok=True)
 
    def download_model(self):
        self.model_path_onnx.parent.mkdir(parents=True, exist_ok=True)

        if not self.model_path_onnx.is_file():
            print(f"El modelo ONNX no existe en: {self.model_path_onnx}")
            print(f"Descargando desde {self.url_model}...")
            
            response = requests.get(self.url_model, allow_redirects=True)
            with open(self.model_path_onnx, "wb") as file:
                file.write(response.content)
            print(f"Modelo descargado en {self.model_path_onnx}")
    
    def download_dataset(self):
        self.gt_path.mkdir(parents=True, exist_ok=True)
        
        if not self.zip_path.is_file():
            print(f"Descargando dataset desde: {self.url_dataset}")
            response = requests.get(self.url_dataset, allow_redirects=True)
            with open(self.zip_path, "wb") as file:
                file.write(response.content)
        if not any(self.gt_path.iterdir()):
            print(f"Descomprimiendo dataset desde {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.gt_path)
            print(f"Descomprimido en: {self.gt_path}")
            self.merge_folders(self.gt_path)
        self.get_yaml_data()
        self.get_dataset_images()
        
    def get_yaml_data(self):
        self.yaml_path = self.gt_path / Path("data.yaml")
        # Cargar clases desde el archivo YAML
        with self.yaml_path.open("r") as f:
            data = yaml.safe_load(f)
        self.classes = data.get("names", [])
        np.random.seed(42)
        self.classes_color = np.random.uniform(0, 255, size=(len(self.classes), 3)) # Generar colores para cada clase

    def merge_folders(self, gt_path: Path):
        for subdir in list(gt_path.iterdir()):  
            if subdir.is_dir() and subdir != gt_path:  # Asegurar que es una subcarpeta
                # print(f"Fusionando: {subdir}")
                for content in list(subdir.iterdir()):
                    dest_path = gt_path
                    if content.is_dir():  # Fusionar carpetas
                        for sub_content in content.iterdir():
                            shutil.move(str(sub_content), str(dest_path / sub_content.name))
                        content.rmdir()
                    else:
                        shutil.move(str(content), str(dest_path))
                subdir.rmdir()
        # print(f"¡Carpetas fusionadas en '{gt_path}'!")
    
    def get_dataset_images(self):
      self.images_sample = sorted([img for ext in ["*.jpg", "*.jpeg", "*.png"] for img in self.gt_path.glob(ext)])
      # image_path = dir_images_path / Path("Avances-Elite-Toluquilla-II-a-julio-2024_mp4-0021.jpg") # Imagen especifica
      return self.images_sample

class ModelProcessor:
    def __init__(self, model_files, **kwargs):
        self.model_files = model_files
        self.debug = kwargs.get("debug", False)
        self.pkl_path = kwargs.get("pkl_path", "")

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{self.model_files.classes[class_id]} ({confidence:.2f})'
        color = self.model_files.classes_color[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, self.target_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        chw_image = np.transpose(normalized_image, (2, 0, 1))
        batched_image = np.expand_dims(chw_image, axis=0)

        new_inputs_numpy = {"images": batched_image}

        if self.debug:
            for k, v in new_inputs_numpy.items():
                print(f"{k}: {v.shape}")

        inputs = {k: Tensor(v, device="NPY").realize() for k, v in new_inputs_numpy.items()}
        return inputs

    def postprocess_image(self, output_tensor, original_image):
        timer = Timer()
        timer.start()

        height, width, _ = original_image.shape
        scale = max(height, width) / 640
        score_threshold = 0.45

        resized_image = cv2.resize(original_image, (640, 480))
        fase1_timer = timer.stop()

        timer.start()
        output_tensor = output_tensor.numpy()
        fase2_timer = timer.stop()

        if self.debug:
            print("output_tensor:", output_tensor)

        timer.start()
        output_tensor = np.transpose(output_tensor, (0, 2, 1))
        rows = output_tensor.shape[1]

        if self.debug:
            print(f"Forma transpuesta: {output_tensor.shape}")

        boxes, scores, class_ids = [], [], []
        for i in range(rows):
            classes_scores = output_tensor[0][i][4:]
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= score_threshold:
                box = [
                    output_tensor[0][i][0] - 0.5 * output_tensor[0][i][2],
                    output_tensor[0][i][1] - 0.5 * output_tensor[0][i][3],
                    output_tensor[0][i][2],
                    output_tensor[0][i][3]
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        nms_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, 0.45, 0.5)
        detections = []

        for i in range(len(nms_boxes)):
            index = nms_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': self.model_files.classes[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale
            }
            detections.append(detection)

            x, y, x_plus_w, y_plus_h = (
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale)
            )
            self.draw_bounding_box(resized_image, class_ids[index], scores[index], x, y, x_plus_w, y_plus_h)

        fase3_timer = timer.stop()

        print(f"Post-Inferencia: fase 1: {fase1_timer}ms, fase 2: {fase2_timer}ms, fase 3: {fase3_timer}ms")
        return resized_image

    def compile_model(self, images_sample):
        # Device.DEFAULT = getenv("DEVICE", "CPU")
        print("Pickle: ", self.pkl_path)
        print("Dispositivo:", Device.DEFAULT)

        run_onnx = OnnxRunner(self.onnx_model)
        run_onnx_jit = TinyJit(
            lambda **kwargs: next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
            prune=True
        )

        input_types = {
            inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in self.onnx_model.graph.input
        }
        input_types = {k: (np.float32 if v == np.float16 else v) for k, v in input_types.items()}

        print("Input types:", input_types)

        for i in range(3):
            GlobalCounters.reset()
            print(f"Ejecutando prueba {i}")

            image = cv2.imread(images_sample[i])
            if image is None:
                raise ValueError("No se pudo cargar la imagen.")

            model_input = self.preprocess_image(image)

            with Context(DEBUG=2 if i == 2 else 1):
                ret = run_onnx_jit(**model_input).numpy()

            if i == 1:
                test_val = np.copy(ret)

        print(f"Captured {len(run_onnx_jit.captured.jit_cache)} kernels")

        try:
            np.testing.assert_allclose(test_val, ret, rtol=1e-3, atol=1e-3)
            print("JIT validado correctamente")
        except AssertionError as e:
            print("Fallo en validación JIT:", e)

        kernel_count = sum(1 for ei in run_onnx_jit.captured.jit_cache if isinstance(ei.prg, CompiledRunner))
        print(f"Kernels compilados: {kernel_count}")

        with open(self.pkl_path, "wb") as f:
            pickle.dump(run_onnx_jit, f)

        print(f"Modelo ONNX: {os.path.getsize(self.model_files.model_path_onnx) / 1e6:.2f} MB")
        print(f"Modelo PKL: {os.path.getsize(self.pkl_path) / 1e6:.2f} MB")
        print("**** Compilación finalizada ****")

        return run_onnx_jit

    def load_onnx(self, onnx_path):
      self.onnx_model = onnx.load(open(onnx_path, "rb"))
      self.input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in self.onnx_model.graph.input}
      self.target_size = tuple(reversed(self.input_shapes['images'][2:]))

if __name__ == "__main__":
  os.chdir("/tmp")
  model_files = ModelDownloader(args.url_model, args.url_dataset)
  model_files.download_model()
  model_files.download_dataset()
  
  tinygrad_model = ModelProcessor(model_files)
  tinygrad_model.load_onnx(model_files.model_path_onnx)
  
  print("Classes: ", model_files.classes) if debug else None

  if not args.model_path_pkl:
    tinygrad_model.pkl_path = model_files.model_path_onnx.parent / "best.pkl"
    print(tinygrad_model.pkl_path)
  else:
    tinygrad_model.pkl_path = args.model_path_pkl
    
  if not Path(tinygrad_model.pkl_path).is_file():
    print(f"El archivo '{tinygrad_model.pkl_path}' no existe.")
    print(f"Compilando...")
    run_onnx_jit = tinygrad_model.compile_model(model_files.images_sample[:3])
  else: 
      print(f"Abriendo modelo compilado en {tinygrad_model.pkl_path}")
      with open(tinygrad_model.pkl_path, "rb") as f:
          run_onnx_jit = pickle.load(f)
    
  print(run_onnx_jit) if debug else None
  # exit()

  # images_sample = ["Avances-Elite-Toluquilla-II-a-julio-2024_mp4-0021.jpg"]
  timer = Timer()
  acum_time = []
  for image_path in model_files.images_sample[:10]:
    timer.start()
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError("No se pudo cargar la imagen.")
    
    # Pre-procesar la imagen a inputs que espera en modelo tinyjit
    model_input = tinygrad_model.preprocess_image(original_image)
    preinferencia_time = timer.stop()
    
    # Ejecutar la inferencia
    timer.start()
    output_tensor = run_onnx_jit(**model_input)  
    inferencia_time = timer.stop()
    
    print("Raw tensor: ", output_tensor) if debug else None
    
    # Post-procesar la imagen con las detecciones
    timer.start()
    postprocessed_image = tinygrad_model.postprocess_image(output_tensor, original_image)
    postinferencia_time = timer.stop()
    
    # Imprimir velocidad del modelo
    print(f"Pre-inferencia: {preinferencia_time}ms, Inferencia: {inferencia_time}ms, Post-inferencia: {postinferencia_time}ms")
    
    total_time = preinferencia_time + inferencia_time + postinferencia_time
    acum_time.append(total_time)
    
    # Mostrar la imagen con las detecciones
    if args.imshow:
      cv2.imshow("Tinygrad model", postprocessed_image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cv2.destroyAllWindows()
  print(f"\tMedian: {np.median(acum_time)}ms")

# # Guardar o mostrar la imagen con las detecciones
# output_path = "output_image.jpg"
# cv2.imwrite(output_path, output_image)
# print(f"Imagen procesada guardada en: {output_path}")