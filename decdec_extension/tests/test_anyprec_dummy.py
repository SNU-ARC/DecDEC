import torch
import decdec_ext
import nvtx

def run_dummy_anyprec_profile(dummy_sm_values, dummy_iters=1 << 16, bitwidth=3):
    in_ch = 1024 * 32
    out_ch = 1024 * 32

    torch.manual_seed(0)

    x_fp32 = torch.rand((1, 1, in_ch), dtype=torch.float32).cuda()
    qweight_int = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (bitwidth, out_ch, in_ch // 32),
        dtype=torch.int32
    ).cuda()
    lut_fp32 = torch.rand(out_ch, 2 ** bitwidth, dtype=torch.float32).cuda()

    dtype = torch.float16
    x = x_fp32.to(dtype)
    lut = lut_fp32.to(dtype)
    output = torch.zeros((1, 1, out_ch), dtype=dtype).cuda()

    # Do 10 warmup iterations
    with nvtx.annotate("Warmup", color="green"):
        for _ in range(10):
            decdec_ext.dummy_anyprec(x, output, qweight_int, lut, bitwidth, 0, dummy_iters)
        torch.cuda.synchronize()

    for sm in dummy_sm_values:
        tag = f"dummy_sm={sm}"
        print(f"Launching: {tag}")
        with nvtx.annotate(tag, color="blue"):
            decdec_ext.dummy_anyprec(x, output, qweight_int, lut, bitwidth, sm, dummy_iters)
            torch.cuda.synchronize()

if __name__ == "__main__":
    dummy_sms = list(range(0, 64, 1))  # 0 to 64 inclusive
    run_dummy_anyprec_profile(dummy_sms)
