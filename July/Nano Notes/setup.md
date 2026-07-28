# Steps

- python3.10 -m venv nano_env
- source nano_env/bin/activate

## Set up QNN SDK
- export QNN_SDK_ROOT=/path/to/SDK
- source ${QNN_SDK_ROOT}/bin/envsetup.sh
- ${QNN_SDK_ROOT}/bin/check-python-dependency

## Install dependencies
- pip install ./nanoV4_gg/transformers-5.6.0.dev0-py3-none-any.whl
- pip install torch numpy scipy tiktoken protobuf peft --index-url https://pypi.org/simple
- python -m pip install pillow
- pip install -r ./NanoV4/requirements.txt

## create symlink to google model folder as (data)
- ln -s /prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/Nano/nanoV4_gg data

## commnd to run :
 python qmain.py -mid ./data/nano_v4_fast -mnm fast -out ./output_debug --arn 521 --context_length 15527 --sliding_window_length 1792 --vision_soft_tokens 280 --calibration_size 1 --modality vision --vision_folder vision --text_folder text --mtp_folder mtp
 
