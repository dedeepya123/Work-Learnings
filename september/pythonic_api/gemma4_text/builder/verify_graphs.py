
import json
import os
import subprocess
import sys
import tempfile

QAIRT_SDK_ROOT = os.path.join(os.getcwd(), "qairt/2.48.40.260702")
CONTEXT_BINARY_UTILITY = os.path.join(QAIRT_SDK_ROOT, "bin/x86_64-linux-clang/qnn-context-binary-utility")
CONTEXT_BINARY_UTILITY_LIB_DIR = os.path.join(QAIRT_SDK_ROOT, "lib/x86_64-linux-clang")


def dump_graph_names(context_binary_path: str) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CONTEXT_BINARY_UTILITY_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")

    subprocess.run(
        [
            CONTEXT_BINARY_UTILITY,
            f"--context_binary={context_binary_path}",
            f"--json_file={json_path}",
        ],
        env=env,
        check=True,
    )

    with open(json_path) as f:
        info = json.load(f)
    os.remove(json_path)

    return [g["info"]["graphName"] for g in info["info"]["graphs"]]


def verify_container(container_dir: str) -> None:
    models_dir = os.path.join(container_dir, "models")
    split_dirs = sorted(d for d in os.listdir(models_dir) if d.startswith("split_"))

    for split in split_dirs:
        bin_path = os.path.join(models_dir, split, "model.bin")
        graph_names = dump_graph_names(bin_path)
        print(f"{split}:")
        for name in graph_names:
            print(f"  {name}")


if __name__ == "__main__":
    verify_container(sys.argv[1] if len(sys.argv) > 1 else "nano/llm_container_prefix_decode")
