# 双向诚实门禁（构建期运行；由 Dockerfile 以 `RUN python /tmp/gate_script.py` 调用）
import importlib.util as iu
import subprocess

# A) claimed-usable：镜像声称可用的包，必须真的能 import
MODS = [
    ("qwen_vl_utils", "qwen-vl-utils"), ("qwen_omni_utils", "qwen-omni-utils"),
    ("decord", "decord"), ("av", "av"), ("librosa", "librosa"),
    ("soundfile", "soundfile"), ("torchaudio", "torchaudio"), ("timm", "timm"),
    ("cv2", "opencv-python"), ("diffusers", "diffusers"), ("deepspeed", "deepspeed"),
    ("liger_kernel", "liger-kernel"), ("fla", "flash-linear-attention"),
    ("tilelang", "tilelang"), ("z3", "z3-solver"), ("vllm", "vllm"), ("ray", "ray"),
    ("cuda", "cuda-tile"), ("cutlass", "nvidia-cutlass-dsl-libs-base"),
    ("torch_c_dlpack_ext", "torch-c-dlpack-ext"), ("triton", "tokenspeed-triton"),
    ("opentelemetry.semconv", "opentelemetry-semantic-conventions"),
    ("opentelemetry.proto", "opentelemetry-proto"),
    ("opentelemetry.exporter.otlp.proto.common", "opentelemetry-exporter-otlp-proto-common"),
    ("opentelemetry.exporter.otlp.proto.grpc", "opentelemetry-exporter-otlp-proto-grpc"),
    ("opentelemetry.exporter.otlp.proto.http", "opentelemetry-exporter-otlp-proto-http"),
    ("google.protobuf", "googleapis-common-protos"), ("pynvml", "nvidia-ml-py"),
    ("elftools", "pyelftools"), ("tabulate", "tabulate"), ("astor", "astor"),
    ("supervisor", "supervisor"), ("interegular", "interegular"),
    ("docstring_parser", "docstring-parser"), ("distro", "distro"),
    ("loguru", "loguru"), ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"), ("jwt", "PyJWT"), ("multipart", "python-multipart"),
    ("sse_starlette", "sse-starlette"), ("ml_dtypes", "ml-dtypes"),
]

# B) installed-but-known-unusable：装了但已知不可用的后端，必须断言 import 失败
UNUSABLE = [
    (
        "megatron-backend",
        "from swift.megatron.arguments import MegatronSftArguments",
        "transformer_engine",
        "megatron_backend UNAVAILABLE (missing transformer_engine)",
    ),
]

problems = []
for mod, dist in MODS:
    if iu.find_spec(mod) is None:
        problems.append("module not found: %s (dist %s)" % (mod, dist))
        continue
    try:
        __import__(mod)
        print("OK   import %-45s (dist %s)" % (mod, dist))
    except Exception as exc:
        problems.append("import failed: %s (dist %s): %r" % (mod, dist, exc))
        print("FAIL import %-45s (dist %s): %r" % (mod, dist, exc))

caps = []
for label, stmt, sub, cap in UNUSABLE:
    r = subprocess.run(["python", "-c", stmt + "; print('UNEXPECTED_IMPORT_OK')"],
                       capture_output=True, text=True, timeout=600)
    blob = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "UNEXPECTED_IMPORT_OK" in (r.stdout or ""):
        problems.append("known-unusable " + label + " IMPORTED OK -- capability declaration is stale")
        print("FAIL unusable " + label + ": imported OK (must fail)")
    elif sub not in blob:
        problems.append("known-unusable " + label + " failed for wrong reason (want " + sub + ")")
        print("FAIL unusable " + label + ": wrong failure reason")
    else:
        caps.append(cap)
        print("OK   unusable %s -> %s" % (label, cap))

if problems:
    raise SystemExit("DISHONEST GATE FAILED: " + " | ".join(problems))

with open("/opt/graspo-image/VERSION", "a") as fh:
    for line in caps:
        fh.write(line + "\n")
print("HONEST GATE OK: %d claimed-usable modules import; capability lines: %s" % (len(MODS), caps))
