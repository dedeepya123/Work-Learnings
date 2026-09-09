
import os

from qairt.api.configs.device import Device, DevicePlatformType
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor

CONTAINER_PATH = os.path.join(os.getcwd(), "nano/llm_container_prefix_decode_new")


ANDROID_SERIAL = os.environ.get("ANDROID_SERIAL")
ANDROID_HOSTNAME = os.environ.get("ANDROID_HOSTNAME")


def load_container(container_path: str = CONTAINER_PATH) -> LLMContainer:
    return LLMContainer.load(container_path)


def make_executor(container: LLMContainer, device: Device | None = None, clean_up: bool = True) -> T2TExecutor:
    return T2TExecutor(
        container.models,
        container.gen_ai_config,
        container.backend,
        device=device,
        backend_extensions_config=container.backend_extensions_config,
        clean_up=clean_up,
    )


def generate(
    prompt: str,
    container_path: str = CONTAINER_PATH,
    device: Device | None = None,
    clean_up: bool = True,
) -> str:
    container = load_container(container_path)
    executor = make_executor(container, device=device, clean_up=clean_up)
    with executor:
        result = executor.generate(prompt)
    return result.generated_text


def android_device(serial: str = ANDROID_SERIAL, hostname: str | None = ANDROID_HOSTNAME) -> Device:
    if not serial:
        raise ValueError("No Android serial set -- pass serial= or set $ANDROID_SERIAL.")
    identifier = f"{serial}@{hostname}" if hostname else serial
    return Device(type=DevicePlatformType.ANDROID, identifier=identifier)


if __name__ == "__main__":

    device = android_device() if ANDROID_SERIAL else None
    text = generate("What is the capital of France?", device=device)
    print("generated:", text)
