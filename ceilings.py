import argparse, torch, time

parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=4096)
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--K", type=int, default=4096)
args = parser.parse_args()

assert torch.cuda.is_available()
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)

def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters

# Ceiling 1: memory bandwidth via large-tensor copy
n = 1 << 28  # ~268M floats = 1GiB
src = torch.randn(n, device=device, dtype=torch.float32)
dst = torch.empty_like(src)

def copy_fn():
    dst.copy_(src)

t_copy = timed(copy_fn)
bytes_moved = 2 * n * 4  # read src + write dst, 4 bytes/float32
bandwidth_gbps = bytes_moved / t_copy / 1e9
print(f"[{gpu_name}] Measured bandwidth: {bandwidth_gbps:.1f} GB/s")

# Ceiling 2: FP32 throughput via large cuBLAS matmul
torch.backends.cuda.matmul.allow_tf32 = False  # force real FP32, not TF32
M, N, K = args.M, args.N, args.K
a = torch.randn(M, K, device=device, dtype=torch.float32)
b = torch.randn(K, N, device=device, dtype=torch.float32)

def matmul_fn():
    torch.matmul(a, b)

t_matmul = timed(matmul_fn)
flops = 2 * M * N * K
tflops = flops / t_matmul / 1e12
print(f"[{gpu_name}] Measured FP32 throughput: {tflops:.2f} TFLOPS")
ridge = tflops * 1000 / bandwidth_gbps  # GFLOPS / (GB/s) = FLOP/byte
print(f"Ridge point: {tflops*1000:.0f} GFLOPS / {bandwidth_gbps:.1f} GB/s = {ridge:.2f} FLOP/byte")