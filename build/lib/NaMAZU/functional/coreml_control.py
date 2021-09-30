import json
from os.path import splitext


def drop_negative(json_path: str, save: bool = True, out_path: str = "") -> int:
    """Load annotation file for CoreML API and drop negative samples.

    Return value is the number of samples successfully removed.

    Args:
        json_path (str): Path to the annotation file.
        save (bool): Whether to save the mutated annotation file.
        out_path (str, optional): Output path of json. Defaults to use f"droped_{json_path}"

    Returns:
        int: Number of successful deletion.
    """
    annotation = json.load(open(json_path))
    ano_to_remove = []
    for i, ano in enumerate(annotation):
        bbs = ano["annotations"]
        for bb in bbs:
            print(bb)
            if bb["label"] == "Negative":
                bbs.remove(bb)
                ano_to_remove.append(i)
                break

    count = 0
    for i in reversed(ano_to_remove):
        annotation.pop(i)
        count += 1

    if save:
        if out_path == "":
            out_path = f"droped_{json_path}"
        elif splitext(out_path)[1] != ".json":
            out_path += ".json"
        with open(out_path, "w") as fout:
            json.dump(annotation, fout)

    return count
