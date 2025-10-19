# quick_check_flash_attn.py
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDAが無効です。GPU/CUDAのセットアップを確認してください。")

device = torch.device("cuda")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {name}  Compute Capability: {cc[0]}.{cc[1]}")

try:
    import flash_attn
    from flash_attn import __version__ as fa_ver
    print("flash_attn version:", fa_ver)
except Exception as e:
    print("flash_attn の import に失敗:", repr(e))
    sys.exit(1)

# 推奨チェック
major_minor = cc[0] + cc[1]/10.0
if major_minor < 8.0:
    print("警告: このGPUのCCは < 8.0 です。FlashAttention v2 は性能/対応が限定される可能性があります。")

# dtype確認（fp16/bf16が基本）
print("AMP bf16 available:", torch.cuda.is_bf16_supported())
print("AMP fp16 available:", torch.cuda.is_available())
print("OK: 基本的な環境チェックは通過しました。")
