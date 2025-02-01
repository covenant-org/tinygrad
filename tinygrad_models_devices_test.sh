#!/bin/bash

# Definir las variables de entorno
export DEBUG=0
export FLOAT16=1
export JIT=2
export IMAGE=1

# Rutas de ejecucion
PYTHON_EXEC="tinygrad_env/bin/python"
SCRIPT_PATH="examples/yolov8-onnx_jit.py"
JIT_MODEL="models/yoloV8-Medium-NucleaV9/best.pkl"

# Extraer backends disponibles
DEVICES_STR=$($PYTHON_EXEC -c "from tinygrad import Device; print(' '.join(Device.get_available_devices()))")
read -ra DEVICES <<< "$DEVICES_STR"

for DEVICE in "${DEVICES[@]}"; do
    echo "Ejecutando dispositivo: $DEVICE"
    $PYTHON_EXEC $SCRIPT_PATH --device "$DEVICE" --imshow "False"
    if [ $? -ne 0 ]; then
        echo "Error ejecutando el modelo: $MODEL_URL"
        exit 1
    fi
    echo "Eliminando $JIT_MODEL para siguiente prueba"
    rm $JIT_MODEL
done

echo "Todos los modelos han sido procesados correctamente."
