import os
import re
import argparse
import torch
from safetensors.torch import load_file


def tensor_stats(W: torch.Tensor):
    W = W.float()
    fro = torch.linalg.norm(W).item()                 # Frobenius
    spec = torch.linalg.norm(W, ord=2).item()         # spectral norm
    mean_abs = W.abs().mean().item()
    max_abs = W.abs().max().item()

    # Added: concentration / sharpness indicators (minimal extra compute)
    # stable rank = ||W||_F^2 / ||W||_2^2 (smaller => energy more concentrated)
    stable_rank = (fro * fro) / (spec * spec + 1e-12)
    # spec/fro (larger => more concentrated)
    spec_over_fro = spec / (fro + 1e-12)

    return fro, spec, mean_abs, max_abs, stable_rank, spec_over_fro



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="directory containing model.safetensors")
    ap.add_argument("--file", type=str, default="model.safetensors", help="weights filename (default: model.safetensors)")
    ap.add_argument("--pattern", type=str, default=None,
                    help="regex to match weight name, e.g. 'query_transform.*weight'")
    ap.add_argument("--scale", type=float, default=None,
                help="optional: query_transform_scale used in residual z + scale*Wz; prints residual upper bound assuming ||z||=1")
    args = ap.parse_args()

    path = os.path.join(args.model_dir, args.file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    sd = load_file(path)  # dict[name] -> tensor (on CPU)
    print(f"[Loaded] {path}")
    print(f"[Keys] {len(sd)} tensors in state_dict")

    default_patterns = [
        r"query_transform.*weight",
        r"transform.*weight",
        r".*projection.*weight",
        r".*proj.*weight",
        r".*linear.*weight",
    ]
    patterns = [args.pattern] if args.pattern else default_patterns

    matched = []
    for name, t in sd.items():
        if not torch.is_tensor(t):
            continue
        if not name.endswith("weight"):
            continue
        for pat in patterns:
            if re.search(pat, name):
                matched.append((name, t))
                break

    if not matched:
        print("[No match] No candidate weight tensors matched your patterns.")
        print("Tip: print all keys containing 'weight' to discover the exact name.")
        # 打印一些提示
        keys = [k for k in sd.keys() if k.endswith("weight")]
        print(f"[Hint] state_dict has {len(keys)} weight keys. Showing up to 50:")
        for k in keys[:50]:
            print(" ", k)
        return

    print(f"[Matched] {len(matched)} candidate weight tensors:\n")
    for name, W in matched:
        if W.ndim != 2:
            continue
        fro, spec, mean_abs, max_abs, stable_rank, spec_over_fro = tensor_stats(W)
        print(f"- {name}")
        print(f"  shape        : {tuple(W.shape)}")
        print(f"  fro_norm     : {fro:.6f}")
        print(f"  spec_norm    : {spec:.6f}")
        print(f"  spec/fro     : {spec_over_fro:.6f}   (larger => more concentrated/sharper)")
        print(f"  stable_rank  : {stable_rank:.3f}     (smaller => sharper/less isotropic)")
        print(f"  mean|w|      : {mean_abs:.6f}")
        print(f"  max|w|       : {max_abs:.6f}")

        if args.scale is not None:
            # If embeddings are normalized (||z||=1), then ||scale*Wz|| <= |scale|*||W||_2
            print(f"  residual_ub  : <= {abs(args.scale) * spec:.6f}  (assuming ||z||=1)")

        print()


    print("[Optional] Distance to Identity (only for square matrices):\n")
    for name, W in matched:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            continue
        Wf = W.float()
        I = torch.eye(W.shape[0])
        dist_I = torch.linalg.norm(Wf - I).item()
        dist_0 = torch.linalg.norm(Wf).item()
        print(f"- {name}")
        print(f"  ||W - I||_F : {dist_I:.6f}")
        print(f"  ||W||_F     : {dist_0:.6f}")
        print()


if __name__ == "__main__":
    main()
