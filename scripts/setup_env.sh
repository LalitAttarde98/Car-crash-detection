
pip install torch==2.1.2+cu118
pip install supervision
pip install Pillow
pip install scipy
pip install av
pip install rfdetr==1.5.0.rc1
pip install peft==0.7.1
pip install transformers==4.50.0.dev0
pip install lap==0.5.13
pip install cython-bbox==0.1.5
pip install optuna==4.8.0
pip install accelerate==1.13.0
pip install git+https://github.com/huggingface/transformers@v4.49.0-SmolVLM-2
pip install num2words timm==0.9.0


# Fix `dinov2_with_windowed_attn.py`
FILE_PATH=$(find ~/.local/lib -name "dinov2_with_windowed_attn.py")
if [ -f "$FILE_PATH" ]; then
    echo "Found dinov2_with_windowed_attn.py at $FILE_PATH"
    sed -i.bak '/torch_int,/d' "$FILE_PATH"
    sed -i '/logger = logging.get_logger(__name__)/a \
import torch\
\
def torch_int(x):\
    if torch.jit.is_tracing():\
        return x.to(torch.int64)\
    return int(x)\
' "$FILE_PATH"
    echo "Patched dinov2_with_windowed_attn.py"
else
    echo "Warning: dinov2_with_windowed_attn.py not found. The script might fail."
fi
