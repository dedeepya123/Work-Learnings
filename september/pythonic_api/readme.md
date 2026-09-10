# NanoV4 pythonic-api 

This folder contains the Nano model preparation and runtime flow for the Gemma4 text pipeline using pythonic apis.

The project is organized around three main stages:

- model preparation and validation in `main.py`
- container creation in the builder code under `models/gemma4_text/builder/`
- model execution in the runtime code under `models/gemma4_text/runtime/`

## Project layout

```text
pythonic_api/nano/
├── main.py
├── models/
│   └── gemma4_text/
│       ├── adapt.py
│       ├── checkpoint.py
│       ├── qc_config.py
│       ├── quantize.py
│       ├── verify.py
│       ├── builder/
│       │   ├── container.py
│       │   ├── prefix_decode.py
│       │   └── ...
│       └── runtime/
│           ├── executor.py
│           └── ...
└── README.md
```

## 1. Setup

Follow the Qualcomm QAIRT setup guide first:

- QAIRT Development Python API setup: https://docs.qualcomm.com/doc/80-87189-2/topic/setup.html?product=1601111740009302

After the SDK environment is set up, activate the QAIRT Python environment and install the local Transformers wheel used for nanov4 project:

```bash
source /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/qairt_env_nano/bin/activate
pip install /path/to/transformers-5.3.0.dev0-py3-none-any.whl
```

After installation, verify the environment is usable:

```bash
python -c "import transformers; print(transformers.__version__)"
```

Then continue with the Nano commands below.

## 2. Main entry point

The top-level script at [pythonic_api/nano/main.py](main.py) is the main preparation pipeline.

What it does:

1. loads the model checkpoint from the configured Nano checkpoint directory
2. builds the QAIRT quantization config
3. reauthorizes the model for the targeted architecture
4. verifies the adapted model structure
5. prepares the quantized model for container export

Run it with:

```bash
cd /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/nano
source /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/qairt_env_nano/bin/activate
python main.py
```

## 3. Builder flow: prefix + decode container

The container-building logic is implemented in [pythonic_api/nano/models/gemma4_text/builder/prefix_decode.py](models/gemma4_text/builder/prefix_decode.py).

This script builds a prefix+decode container using the QAIRT GenAI builder flow:

- loads the checkpoint
- creates the QC config
- resizes the model for the prefix ARN and decode ARN
- attaches the decode graph to the prefix graph
- enables weight sharing
- saves the final container to disk

Typical build command:

```bash
cd /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/nano
source /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/qairt_env_nano/bin/activate
python models/gemma4_text/builder/prefix_decode.py
```

The output container path used by the script is:

```text
pythonic_api/nano/nano/llm_container_prefix_decode_new
```

## 4. Runtime execution

The runtime executor is defined in [pythonic_api/nano/models/gemma4_text/runtime/executor.py](models/gemma4_text/runtime/executor.py).

This script:

1. loads the saved LLM container
2. creates a `T2TExecutor`
3. executes generation with the provided prompt
4. prints the generated text

Example usage:

```bash
cd /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/nano
source /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/pythonic_api/qairt_env_nano/bin/activate
python models/gemma4_text/runtime/executor.py
```

If an Android device is configured through `ANDROID_SERIAL`, the runtime will use that device automatically, otherwise it will use the local/default execution path.

## 5. Typical pipeline

A normal end-to-end workflow is:

```bash
python main.py
python models/gemma4_text/builder/prefix_decode.py
python models/gemma4_text/runtime/executor.py
```

This produces the standard flow:

```text
checkpoint -> reauthor + QC config -> quantized model -> prefix/decode container -> executor generation
```

## 6. Notes

- The project assumes the QAIRT Python environment is active.
- Some paths are hard-coded to local workspace directories
