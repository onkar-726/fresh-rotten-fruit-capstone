"""Check the GPUs before training (run this FIRST on the DGX B200).

    python scripts/check_gpu.py            # fails (exit 1) if TensorFlow sees no GPU
    python scripts/check_gpu.py --allow-cpu
"""
import argparse
import sys

from _common import ROOT  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf

    info = tf.sysconfig.get_build_info()
    print("TensorFlow:", tf.__version__, "| built with CUDA:", tf.test.is_built_with_cuda(),
          "| CUDA", info.get("cuda_version"), "| cuDNN", info.get("cudnn_version"))
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TensorFlow: {len(gpus)}")
    for g in gpus:
        try:
            d = tf.config.experimental.get_device_details(g)
            print(f"  {g.name}: {d.get('device_name')}, compute capability {d.get('compute_capability')}")
        except Exception:  # noqa: BLE001
            print(f"  {g.name}")
    if not gpus:
        print("No GPU visible. On the DGX check: nvidia-smi, the container/CUDA version, CUDA_VISIBLE_DEVICES.")
        sys.exit(0 if a.allow_cpu else 1)

    ok = True
    for i in range(len(gpus)):                        # 1. every GPU can run a matmul
        try:
            with tf.device(f"/GPU:{i}"):
                x = tf.random.normal((2048, 2048))
                float(tf.reduce_sum(tf.matmul(x, x)))
            print(f"  GPU:{i} matmul OK")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  GPU:{i} matmul FAILED: {e}")
    if len(gpus) > 1:                                 # 2. multi-GPU all-reduce works
        try:
            s = tf.distribute.MirroredStrategy()
            r = s.run(lambda: tf.ones(()))
            print("  MirroredStrategy all-reduce OK, replicas:", s.num_replicas_in_sync,
                  "sum =", float(s.reduce(tf.distribute.ReduceOp.SUM, r, axis=None)))
        except Exception as e:  # noqa: BLE001
            ok = False
            print("  MirroredStrategy FAILED:", e)
    try:                                              # 3. bfloat16 works
        with tf.device("/GPU:0"):
            b = tf.cast(tf.random.normal((256, 256)), tf.bfloat16)
            float(tf.cast(tf.reduce_sum(tf.matmul(b, b)), tf.float32))
        print("  bfloat16 matmul OK")
    except Exception as e:  # noqa: BLE001
        print("  bfloat16 matmul failed (use --mixed-precision off):", e)
    print("GPU CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
