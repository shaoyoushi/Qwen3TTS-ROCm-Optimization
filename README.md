# 🚀 Qwen3-TTS ROCm 异构加速部署方案 (AMD Strix Halo 专版)

本项目旨在为 AMD 最新架构芯片（特别是基于 **Strix Halo / gfx1151** 架构的 Ryzen AI Max+ 395 等 APU）提供极限优化的 Qwen3-TTS 容器化部署方案。

通过深度的底层 API 劫持与异构计算切分，本方案成功绕过了 AMD MIOpen 驱动在卷积运算上的性能瓶颈，实现了 **5 倍以上**的推理加速。

这是和Gemini 3.1 Pro一起鏖战几个小时的成果，参考了之前和Claude一起搞定IndexTTS2的过程。

## 📊 性能表现

> **测试硬件**: AMD Ryzen AI Max+ 395 (16核 Zen5 CPU + 96GB 统一显存)
> **测试任务**: 9 秒高质量语音合成 (Qwen3-TTS 1.7B CustomVoice)

| 运行模式 | 渲染耗时 | 瓶颈原因 |
| :--- | :--- | :--- |
| **官方默认 (纯 GPU)** | `> 70 秒` | MIOpen 缺失 1D-CNN 优化内核，导致 BigVGAN 退化为极慢的 Fallback 算法。 |
| **异构加速 (本项目)** | `~ 13.6 秒` | **性能提升 500%**。GPU 专心跑 LLM 矩阵乘法，CPU 处理卷积解码。 |

---

## 🛠️ 核心优化技术揭秘

为了在不修改官方库源码的前提下实现完美加速，我们在 `Dockerfile` 构建阶段注入了以下“外科手术”级的动态补丁：

### 1. 异构计算与类属性级劫持 (Class-Level Hijacking)
Qwen3-TTS 底层采用 Transformer (LLM) + BigVGAN (Vocoder) 的架构。我们直接揪出底层的 BigVGAN 实体并将其转移至 CPU (`.to('cpu')`)。
为了防止原生代码在数据流动时强行将张量拉回 GPU 导致 `index is on cuda:0` 崩溃，我们直接重写了 Tokenizer 类的设备属性：
```python
TokenizerClass.device = property(lambda self: torch.device('cpu'))
```
这让 Qwen 的原生代码极其乖巧地配合我们将张量稳稳留在 CPU 上，免去了容易翻车的手动数据搬移。

### 2. 暴力清洗视觉依赖防崩溃 (Vision Dependency Purging)
在 AMD gfx1151 架构下，加载 `torchvision` 的 C++ 动态链接库会引发不可捕获的静默崩溃。
由于 `transformers` 的 `loss_utils.py` 无条件引入了大量视觉模型（如 Detr, GroundingDino），我们通过 `sed` 命令对依赖链进行了“无差别物理清洗”，删除了所有计算机视觉相关的引入行，彻底斩断了 C++ Abort 链条。

### 3. 音频后端规避 (Audio Backend Patching)
`torchaudio 2.9.x` 默认调用 `torchcodec` 后端，这在部分 ROCm 环境下会导致崩溃。我们通过动态劫持（Monkey Patching），使用纯净的 `soundfile.write` 整体替换了官方 Demo 中的 `torchaudio.save` 方法。

### 4. 高性能注意力机制 (SDPA Injection)
由于官方原生的 `Flash Attention 2` 暂未完全兼容最新的 ROCm 架构，我们在模型初始化前动态注入了 `attn_implementation="sdpa"`，直接调用 PyTorch 原生的内存高效注意力算法，实现接近 FA2 的推理速度。

---

## 📦 部署指南

### 1. 宿主机环境要求
请确保你的宿主机 Linux 内核已配置以下参数（极其关键）：
* 设置内核参数 `amdgpu.cwsr_enable=0`（防止长时间高负载引发显卡重置）。
* 确保安装了最新的 AMD 专有驱动或处于兼容 ROCm 6.4 的内核环境中。

### 2. 一键构建与启动
将 `Dockerfile`、`docker-compose.yml` 下载至同一目录，执行：
```bash
docker compose up -d --build
```
*注：首次启动时，容器会自动下载 Qwen3-TTS 的模型权重文件（约几 GB），请通过日志关注进度：*
```bash
docker compose logs -f qwen3-rocm-pure
```

### 3. 访问服务
当日志输出 `* Running on local URL:  http://0.0.0.0:8100` 后，打开浏览器访问：
👉 `http://<服务器IP>:8100`

---

## 🧩 目录挂载与数据备份

在 `docker-compose.yml` 中，我们配置了模型权重的本地持久化：
```yaml
volumes:
  - ./hf_cache:/app/hf_cache
```
所有下载的 HuggingFace 模型都将保存在宿主机的 `hf_cache` 目录中。即使重建容器，也无需重新下载数十 GB 的模型文件。
