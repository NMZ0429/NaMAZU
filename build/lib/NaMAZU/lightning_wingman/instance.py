from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


__all__ = ["Result", "Query"]


@dataclass
class Query:
    __query: Dict[str, Any]

    def __str__(self) -> str:
        doc = f"Query: \n"
        for key in self.__query.keys():
            doc += f"      {key}, type {type(self.__query[key])}\n"

        return doc

    def __repr__(self) -> str:
        return f"Query(query={self.__query})"

    def __getitem__(self, key: str):
        return self.__query[key]


@dataclass
class Result:
    __results: Dict[str, Any]

    def __getitem__(self, key):
        return self.__results[key]

    def __setitem__(self, key: str, value):
        self.__results[key] = value

    def __repr__(self) -> str:
        doc = f"Result: \n"
        for key in self.__results.keys():
            doc += f"      {key}, type {type(self.__results[key])}\n"

        return doc
