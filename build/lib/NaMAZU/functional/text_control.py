from typing import List

__all__ = ["search_word_from"]


def search_word_from(
    db: List[str], query: str, split_token: str = "", ingore_case: bool = True
) -> List[str]:
    """Search str qurey from list of str db. If split_token is given,
    split each str in db by split_token and check for exact match.
    If ingore_case is True, ignore case.

    Args:
        db (List[str]): List of str to search from.
        query (str): String to search.
        split_token (str, optional): Token to split string into words. Defaults to "".
        ingore_case (bool, optional):Whether to ignore case. Defaults to True.

    Returns:
        List[str]: List of str that match query.

    Examples:
        >>> search_word_from(["hello world","hello","HelloWorld"], "hellow")
        ['HelloWorld']
        >>> search_word_from(["hello world","hello","HelloWorld"], "hello", " ")
        ['hello world', 'hello']
        >>> search_word_from(["hello world","hello","HelloWorld"], "hello", ingore_case=False)
        ['hello world', 'hello']
    """
    query = query.lower()
    rtn = []
    for x in db:
        x_formatted = x.lower() if ingore_case else x
        if split_token:
            words = x_formatted.split(split_token)
            for y in words:
                if query == y:
                    rtn.append(x)
                    break
        else:
            if query in x_formatted:
                rtn.append(x)

    return rtn
