import hashlib
import importlib
import importlib.util
from pathlib import Path
from transformers import DataCollatorForSeq2Seq


def load_module_from_py_file(py_file: str) -> object:
    """
    Load a .py file as a module. Uses a unique module name derived from the resolved path
    so we never clash with sys.modules['opinionqa_dataset.py'] from an older/wrong load (Colab/Jupyter).
    """
    py_path = Path(py_file).resolve()
    module_name = "_dataset_py_" + hashlib.sha256(str(py_path).encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_custom_dataset(dataset_config, tokenizer, split: str, chat_template: bool=False):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split, chat_template)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e

def get_data_collator(dataset_processer,dataset_config):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_data_collator"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_processer)
    except AttributeError as e:
        print(f"Can not find the custom data_collator in the dataset.py file ({module_path.as_posix()}).")
        print("Using the default data_collator instead.")
        return None

class NoLabelDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, batch):
        batch = super().__call__(batch)
        if "labels" in batch:
            del batch["labels"]
        return batch

def custom_collator_no_labels(dataset_processer, dataset_config):
    return NoLabelDataCollatorForSeq2Seq(dataset_processer)