import zipfile
from pathlib import Path
from typing import List, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


__all__ = [
    "rename_file",
    "collect_file_pathes_by_ext",
    "zip_files",
    "export_list_str_as",
]

#################
# Path Process  #
#################


def rename_file(file_path: str, new_name: str) -> None:
    """Rename a file.

    Args:
        file_path (str): Path of the file to rename.
        new_name (str): New name of the file.
    """
    file = Path(file_path)
    file.rename(file.parent / new_name)


def collect_file_pathes_by_ext(
    target_dir: str, ext_list: List[str]
) -> Dict[str, List[Path]]:
    """Return the list of Path objects of the files in the target_dir that have the specified extensions.

    Args:
        target_dir (str): Directory to search for files.
        ext_list (List[srt]): List of file extensions to search for.

    Returns:
        List[List[Path]]: List of lists of Path objects. The first list contains the files having the first extension
        in the ext_list. The second list contains the files having the second extension in the ext_list and so on.
    """
    target = Path(target_dir)

    rtn = {}
    for ext in ext_list:
        if ext[0] == ".":
            ext = ext.strip(".")
        rtn[ext] = list(target.glob(f"**/*.{ext}"))

    return rtn


def zip_files(target_dir: str, zip_name: str, ext_list: List[str]) -> None:
    """Zip the files in the target_dir that have the specified extensions.

    Args:
        target_dir (str): Directory to search for files.
        zip_name (str): Name of the zip file to create.
        ext_list (List[str]): List of file extensions to search for.
    """
    target = Path(target_dir)
    zip_file = zipfile.ZipFile(target / zip_name, "w", zipfile.ZIP_DEFLATED)
    for ext in ext_list:
        if ext[0] == ".":
            ext = ext.strip(".")
        for file in target.glob(f"**/*.{ext}"):
            zip_file.write(file, file.name)
    zip_file.close()


##################
# Object Process #
##################


def export_list_str_as(
    extension: Literal["csv", "json", "txt"], list_str: List[str], out_name: str = ""
) -> None:
    """Export a list of strings to a file of given extension. The file name is out_name if given.

    Args:
        extension (Literal["csv", "json", "txt"]): Extension of the file to create.
        list_str (List[str]): List of strings to export.
        out_name (str): Name of the file to create. Default is "".
    """
    if out_name == "":
        out_name = "list_str"

    if extension == "csv":
        with open(f"{out_name}.csv", "w") as f:
            for item in list_str:
                f.write(f"{item}\n")
    elif extension == "json":
        with open(f"{out_name}.json", "w") as f:
            f.write(f"{list_str}")
    elif extension == "txt":
        with open(f"{out_name}.txt", "w") as f:
            for item in list_str:
                f.write(f"{item}\n")
    else:
        raise ValueError(
            f"Invalid extension.{extension}. One of ['csv', 'json', 'txt'] is expected."
        )
