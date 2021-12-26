from typing import Union
import requests
from tqdm import tqdm
from pathlib import Path


def download_weight(file_url: str, save_path: Union[str, Path] = "") -> str:
    """Download onnx file to save_path and return the path to the saved file.

    Args:
        file_url (str): url to onnx file
        save_path (Union[str, Path], optional): path to store the fie. Defaults to "".

    Returns:
        str: path to the saved files
    """

    if isinstance(save_path, str):
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)

    output_file = save_path / file_url.split("/")[-1]
    file_size = int(requests.head(file_url).headers["content-length"])

    res = requests.get(file_url, stream=True)
    pbar = tqdm(
        total=file_size, unit="B", unit_scale=True, desc=file_url.split("/")[-1]
    )
    with open(output_file, "wb") as file:
        for chunk in res.iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    return str(output_file)


def print_onnx_information(onnx_path: str) -> None:
    import onnx

    model = onnx.load(onnx_path)
    for info in model.graph.input:  # type: ignore
        print(info.name, end=": ")
        # get type of input tensor
        tensor_type = info.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")
        print()

    for info in model.graph.output:  # type: ignore
        print(info.name, end=": ")
        # get type of output tensor
        tensor_type = info.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")
        print()

