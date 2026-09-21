import torch

print("=" * 50)
print(f"PyTorch 버전     : {torch.__version__}")
print(f"CUDA 빌드 버전   : {torch.version.cuda}")
print(f"CUDA 사용 가능   : {torch.cuda.is_available()}")
print("=" * 50)

if not torch.cuda.is_available():
    print("GPU를 찾지 못했습니다.")
else:
    n = torch.cuda.device_count()
    print(f"감지된 GPU 수    : {n}개")
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        total_mb = prop.total_memory / 1024 ** 2
        print(f"\n[GPU {i}] {prop.name}")
        print(f"  VRAM          : {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
        print(f"  Compute Cap.  : {prop.major}.{prop.minor}")
        print(f"  SM 수         : {prop.multi_processor_count}")

    # 실제 연산 확인
    print("\n--- 연산 테스트 ---")
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print(f"행렬 곱 (1000x1000) : OK  [결과 shape={tuple(z.shape)}, device={z.device}]")

print("=" * 50)
