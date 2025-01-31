#!/bin/bash

# Definir las variables de entorno
export DEBUG=0
export FLOAT16=1
export JIT=2
export IMAGE=1

# Ruta del ejecutable de Python
PYTHON_EXEC="/home/manuelo247/tinygrad/tinygrad_env/bin/python"

# Ruta del script de inferencia
SCRIPT_PATH="/home/manuelo247/tinygrad/examples/yolov8-onnx_jit.py"

# Lista de modelos a descargar
MODELS=(
    "https://github.com/covenant-org/tinygrad/releases/download/yoloV8-Medium-NucleaV9/best.onnx"
    "https://github.com/covenant-org/tinygrad/releases/download/yoloV8-Small-NucleaV9/best.onnx"
    "https://github.com/covenant-org/tinygrad/releases/download/yoloV8-Nano-NucleaV9/best.onnx"
)

# Ejecutar cada modelo secuencialmente
for MODEL_URL in "${MODELS[@]}"; do
    echo "Ejecutando modelo: $MODEL_URL"
    $PYTHON_EXEC $SCRIPT_PATH --url_model "$MODEL_URL" --imshow "False"
    if [ $? -ne 0 ]; then
        echo "Error ejecutando el modelo: $MODEL_URL"
        exit 1
    fi
done

echo "Todos los modelos han sido procesados correctamente."
