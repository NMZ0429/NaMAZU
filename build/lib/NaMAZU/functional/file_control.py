import zipfile
from pathlib import Path
from typing import List, Dict


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
