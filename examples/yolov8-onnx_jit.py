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
from tqdm import tqdm

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

# os.environ.setdefault("CUDA", "1")
# os.environ.setdefault("DEBUG", "4")
# os.environ.setdefault("NOLOCALS", "1") # REDUCE OPTIMIZACION
# os.environ.setdefault("PTX", "1")
# os.environ.setdefault("PROFILE", "1")
# os.environ.setdefault("JIT_BATCH_SIZE", "0")

# Crear el parser
parser = argparse.ArgumentParser(description="Descargar modelo y dataset")
# Agregar argumentos
parser.add_argument("--url_model", type=str, default="https://github.com/covenant-org/tinygrad/releases/download/yoloV8-Nano-NucleaV9/best.onnx", help="URL del modelo ONNX")
parser.add_argument("--url_dataset", type=str, default="https://app.roboflow.com/ds/qnXOxt8VKv?key=1mmF2G81LD", help="URL del dataset")
parser.add_argument("--model_path_pkl", type=str, default="", help="Direccion del pkl")
parser.add_argument("--imshow", type=str, default="False", help="Mostrar imágenes a tiempo real")
args = parser.parse_args()

url_model = args.url_model
url_dataset = args.url_dataset
model_path_pkl = args.model_path_pkl
args.imshow = args.imshow.lower() in ["true", "1", "yes"]

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
        self.load_yaml()
        
    def load_yaml(self):
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

def draw_bounding_box(img, data_model, class_id, confidence, x, y, x_plus_w, y_plus_h):
  label = f'{data_model.classes[class_id]} ({confidence:.2f})'
  color = data_model.classes_color[class_id]
  cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
  cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def preprocess_image(image, target_size=(640, 480)):
  """
    Preprocesa la imagen para que sea compatible con el modelo. Realiza:
    1. Redimensionamiento de la imagen.
    2. Normalización de valores de píxeles.
    3. Cambio de formato de HWC a CHW.
    4. Añadido de dimensión de lote.
    5. Diccionario creado con las entradas procesadas
    
    Args:
        image (np.ndarray): Imagen de entrada en formato HWC.
        target_size (tuple): Tamaño al que redimensionar la imagen (alto, ancho).
        
    Returns:
        dict: Diccionario con las entradas procesadas para el modelo.
  """
  resized_image = cv2.resize(image, target_size)
  normalized_image = resized_image.astype(np.float32) / 255.0
  chw_image = np.transpose(normalized_image, (2, 0, 1))  # De HWC a CHW
  batched_image = np.expand_dims(chw_image, axis=0)  # Añadir dimensión de lote
  
  new_inputs_numpy = {"images": batched_image}
  
  for k, v in new_inputs_numpy.items():
    print(f"{k}: {v.shape}") if debug else None
  
  # Crear las entradas que el modelo espera, y asegurarse de que las claves coincidan con los nombres en el modelo
  inputs = {k: Tensor(v, device="NPY").realize() for k, v in new_inputs_numpy.items()}

  return inputs

def postprocess_image(output_tensor, original_image, data_model):
  timer = Timer()
  timer.start()
  
  # Extraer caracteristicas de imagen
  height, width, _ = original_image.shape
  length = max(height, width)
  scale = length / 640
  
  #
  score_threshold = 0.45
  
  # Redimensionar la imagen a 640x480 para procesamiento
  resized_image = cv2.resize(original_image, (640, 480))
  fase1_timer = timer.stop()
  
  timer.start()
  output_tensor = output_tensor.numpy()
  # print(f"One value tensor: {output_tensor[0, 0, 0]}")
  # output_tensor[0, 0, 0].realize()
  # print(f"One value tensor: {output_tensor[0, 0, 0].item()}")
  fase2_timer = timer.stop()
  print("output_tensor:", output_tensor) if debug else None
  
  timer.start()
  # Procesar tensor de salida y extraer caracteristicas
  batch_size, num_classes_plus_5, num_boxes = output_tensor.shape # Extraer las dimensiones del tensor
  output_tensor = np.transpose(output_tensor, (0, 2, 1)) # Transponer el tensor
  rows = output_tensor.shape[1] # Extraes columbas del tensor
  if debug:
    print("batch_size", batch_size, "num_classes_plus_5", num_classes_plus_5, "num_boxes", num_boxes) 
    print("Forma transpuesta:", output_tensor.shape) 
    print("outputs: ", output_tensor) 
    print("rows: ", rows) 
  
  boxes, scores, class_ids = [], [], []
  for i in range(rows):
      classes_scores = output_tensor[0][i][4:]
      (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
      if maxScore >= score_threshold:
          box = [
                output_tensor[0][i][0] - 0.5 * output_tensor[0][i][2],  # x1
                output_tensor[0][i][1] - 0.5 * output_tensor[0][i][3],  # y1
                output_tensor[0][i][2],  # w
                output_tensor[0][i][3]   # h
          ]
          boxes.append(box)
          scores.append(maxScore)
          class_ids.append(maxClassIndex)

  # Aplicar Non-Maximum Suppression  
  nms_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, 0.45, 0.5)
  
  detections = []
  for i in range(len(nms_boxes)):
      index = nms_boxes[i]
      box = boxes[index]
      detection = {
          'class_id': class_ids[index],
          'class_name': data_model.classes[class_ids[index]],
          'confidence': scores[index],
          'box': box,
          'scale': scale
      }
      detections.append(detection)
      
       # Dibujar las cajas en la imagen redimensionada
      x, y, x_plus_w, y_plus_h = round(box[0] * scale), round(box[1] * scale), round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
      draw_bounding_box(resized_image, data_model, class_ids[index], scores[index], x, y, x_plus_w, y_plus_h)
      
  
  # resized_image = cv2.resize(resized_image, (height, width))
  fase3_timer = timer.stop()
  print("Post-Inferencia: fase 1:", str(fase1_timer) + "ms,", "fase 2:", str(fase2_timer) + "ms,", "fase 3:", str(fase3_timer) + "ms")
  
  return resized_image
    
def compilar_modelo(onnx_model, data_model, images_sample, model_path_pkl):
  run_onnx = OnnxRunner(onnx_model)
  run_onnx_jit = TinyJit(
    lambda **kwargs:
      next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), 
      prune=True
  )

  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  input_types = {k:(np.float32 if v==np.float16 else v) for k,v in input_types.items()}
  print("input shapes: ", input_shapes)
  print("input types: ", input_types)
  
  for i in range(3):
        GlobalCounters.reset()
        print(f"run {i}")
        image = cv2.imread(images_sample[i])
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")

        model_input = preprocess_image(image, target_size=target_size)

        print(model_input)
        # Ejecutar el modelo con las entradas
        with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
            ret = run_onnx_jit(**model_input).numpy()
        
        # Copiar para la validación posterior
        if i == 1:
            test_val = np.copy(ret)

  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  try:
      np.testing.assert_allclose(test_val, ret, rtol=1e-3, atol=1e-3, err_msg="JIT run failed")
      print("jit run validated")
  except AssertionError as e:
      print("Validation failed with differences:")
      print(e)
      
  # checks from compile2
  kernel_count = 0
  read_image_count = 0
  gated_read_image_count = 0
  for ei in run_onnx_jit.captured.jit_cache:
    if isinstance(ei.prg, CompiledRunner):
      kernel_count += 1
      read_image_count += ei.prg.p.src.count("read_image")
      gated_read_image_count += ei.prg.p.src.count("?read_image")
  print(f"{kernel_count=},  {read_image_count=}, {gated_read_image_count=}")
  if (allowed_kernel_count:=getenv("ALLOWED_KERNEL_COUNT", -1)) != -1:
    assert kernel_count <= allowed_kernel_count, f"too many kernels! {kernel_count=}, {allowed_kernel_count=}"
  if (allowed_read_image:=getenv("ALLOWED_READ_IMAGE", -1)) != -1:
    assert read_image_count == allowed_read_image, f"different read_image! {read_image_count=}, {allowed_read_image=}"
  if (allowed_gated_read_image:=getenv("ALLOWED_GATED_READ_IMAGE", -1)) != -1:
    assert gated_read_image_count <= allowed_gated_read_image, f"too many gated read_image! {gated_read_image_count=}, {allowed_gated_read_image=}"

  with open(model_path_pkl, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  mdl_sz = os.path.getsize(data_model.model_path_onnx)
  pkl_sz = os.path.getsize(model_path_pkl)
  print(f"mdl size is {mdl_sz/1e6:.2f}M")
  print(f"pkl size is {pkl_sz/1e6:.2f}M")
  print("**** compile done ****")
  
  return run_onnx_jit

if __name__ == "__main__":
  os.chdir("/tmp")
  data_model = ModelDownloader(url_model, url_dataset)
  data_model.download_model()
  data_model.download_dataset()
  print("Classes: ", data_model.classes) if debug else None
  
  # Directorio de imágenes y lista de imágenes
  dir_images_path = data_model.gt_path
  images_sample = sorted([img for ext in ["*.jpg", "*.jpeg", "*.png"] for img in dir_images_path.glob(ext)])
  image_path = dir_images_path / Path("Avances-Elite-Toluquilla-II-a-julio-2024_mp4-0021.jpg") # Imagen especifica

  onnx_model = onnx.load(open(data_model.model_path_onnx, "rb"))
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  target_size = tuple(reversed(input_shapes['images'][2:]))

  if not model_path_pkl:
    model_path_pkl = data_model.model_path_onnx.parent / "best.pkl"
  if not Path(model_path_pkl).is_file():
    print(f"El archivo '{model_path_pkl}' no existe.")
    
    print(f"Compilando...")
    run_onnx_jit = compilar_modelo(onnx_model, data_model, images_sample, model_path_pkl)
  else: 
      print(f"Abriendo modelo compilado en {model_path_pkl}")
      with open(model_path_pkl, "rb") as f:
          run_onnx_jit = pickle.load(f)
    
  print(run_onnx_jit) if debug else None
  # exit()

  # images_sample = ["Avances-Elite-Toluquilla-II-a-julio-2024_mp4-0021.jpg"]
  timer = Timer()
  for image_path in images_sample:
    timer.start()
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError("No se pudo cargar la imagen.")
    
    # Pre-procesar la imagen a inputs que espera en modelo tinyjit
    model_input = preprocess_image(original_image, target_size=target_size)
    preinferencia_time = timer.stop()
    
    # Ejecutar la inferencia
    timer.start()
    output_tensor = run_onnx_jit(**model_input)  
    inferencia_time = timer.stop()
    
    print("Raw tensor: ", output_tensor) if debug else None
    
    # Post-procesar la imagen con las detecciones
    timer.start()
    postprocessed_image = postprocess_image(output_tensor, original_image, data_model)
    postinferencia_time = timer.stop()
    
    # Imprimir velocidad del modelo
    print(f"Pre-inferencia: {preinferencia_time}ms, Inferencia: {inferencia_time}ms, Post-inferencia: {postinferencia_time}ms")
    
    # Mostrar la imagen con las detecciones
    if args.imshow:
      cv2.imshow("Tinygrad model", postprocessed_image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cv2.destroyAllWindows()

# # Guardar o mostrar la imagen con las detecciones
# output_path = "output_image.jpg"
# cv2.imwrite(output_path, output_image)
# print(f"Imagen procesada guardada en: {output_path}")