FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
ENV DEBIAN_FRONTEND=noninteractive

ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV GPU_MAX_ALLOC_PERCENT=100
ENV GPU_MAX_HEAP_SIZE=100
ENV HSA_ENABLE_SDMA=0
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV MIOPEN_LOG_LEVEL=4
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg sox libsox-dev build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git .

RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

RUN echo 'from importlib.metadata import version' > /tmp/gen_constraints.py \
    && echo 'for pkg in ["torch", "torchaudio"]:' >> /tmp/gen_constraints.py \
    && echo '    v = version(pkg).split("+")[0]' >> /tmp/gen_constraints.py \
    && echo '    print(f"{pkg}=={v}")' >> /tmp/gen_constraints.py \
    && python /tmp/gen_constraints.py > /tmp/torch-constraints.txt

RUN pip install --no-cache-dir setuptools wheel soundfile gradio \
    && pip install --no-cache-dir -c /tmp/torch-constraints.txt -e .

RUN pip uninstall torchvision -y 2>/dev/null || true
RUN LOSS_UTILS=$(python -c "import transformers.loss; import os; print(os.path.join(os.path.dirname(transformers.loss.__file__), 'loss_utils.py'))") \
    && sed -i '/ForSegmentation/d' "$LOSS_UTILS" \
    && sed -i '/ObjectDetection/d' "$LOSS_UTILS" \
    && sed -i '/object_detection/d' "$LOSS_UTILS" \
    && sed -i '/GroundingDino/d' "$LOSS_UTILS" \
    && sed -i '/grounding_dino/d' "$LOSS_UTILS" \
    && sed -i '/Detr/d' "$LOSS_UTILS" \
    && sed -i '/detr/d' "$LOSS_UTILS" \
    && sed -i '/DFine/d' "$LOSS_UTILS" \
    && sed -i '/d_fine/d' "$LOSS_UTILS"

RUN sed -i '/model = AutoModel.from_pretrained/i \        kwargs["attn_implementation"] = "sdpa"' qwen_tts/inference/qwen3_tts_model.py
RUN sed -i 's/with gr.Blocks(theme=theme, css=css) as demo:/with gr.Blocks() as demo:/' qwen_tts/cli/demo.py

# ==========================================================
# 💥 极简纯净版：放弃计时器，直接在类定义级别锁死 CPU 属性！
# ==========================================================
RUN cat << 'EOF' > /app/run_demo.py
import sys
import torch
import torchaudio
import soundfile as sf
from qwen_tts.cli.demo import main
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# 规避 torchaudio 保存崩溃
def patched_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    sf.write(filepath, src.squeeze().cpu().numpy(), sample_rate)
torchaudio.save = patched_torchaudio_save

original_from_pretrained = Qwen3TTSModel.from_pretrained

@classmethod
def patched_from_pretrained(cls, *args, **kwargs):
    kwargs['dtype'] = torch.float16
    print(">> [AMD Hack] 强制要求 FP16 精度...", flush=True)
    wrapper_model = original_from_pretrained(*args, **kwargs)
    
    try:
        target_tokenizer = wrapper_model.model.speech_tokenizer
        TokenizerClass = target_tokenizer.__class__
        
        # 1. 揪出底层的 BigVGAN 物理实体并转移至 CPU
        real_vocoder = target_tokenizer.model if hasattr(target_tokenizer, 'model') else target_tokenizer
        print(f">> [AMD Hack] 正在将 {type(real_vocoder).__name__} 发配至 16 核 CPU...", flush=True)
        real_vocoder.to('cpu').float()
        
        # 2. 终极大招：直接重写该类的 device 属性！
        # 这样无论内部代码怎么调用 self.device，永远只会返回 cpu，彻底掐断数据回流 GPU 的可能！
        TokenizerClass.device = property(lambda self: torch.device('cpu'))
        
        # 3. 在类级别（Class Level）拦截输入数据，确保类型绝对干净
        if hasattr(TokenizerClass, 'decode'):
            orig_decode = TokenizerClass.decode
            def new_decode(self, *d_args, **d_kwargs):
                def _to_cpu(x):
                    if isinstance(x, torch.Tensor): return x.cpu()
                    if isinstance(x, tuple): return tuple(_to_cpu(i) for i in x)
                    if isinstance(x, list): return [_to_cpu(i) for i in x]
                    if isinstance(x, dict): return {k: _to_cpu(v) for k, v in x.items()}
                    return x
                return orig_decode(self, *_to_cpu(d_args), **_to_cpu(d_kwargs))
            TokenizerClass.decode = new_decode
            
        print(">> [AMD Hack] 核心手术完成！纯净切分模式已启动！", flush=True)
    except Exception as e:
        print(f">> [AMD Hack] 警告: 异构切分失败: {e}", flush=True)
            
    return wrapper_model

Qwen3TTSModel.from_pretrained = patched_from_pretrained

if __name__ == "__main__":
    sys.argv[0] = "qwen-tts-demo"
    sys.exit(main())
EOF

CMD ["python3", "/app/run_demo.py", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "--ip", "0.0.0.0", "--port", "8100"]
