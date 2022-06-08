from typing import Dict, List, Optional


class Trie:
    def __init__(self):
        self.__root: TrieNode = TrieNode()

    def search(self, key) -> bool:
        return self.__root.search(key)

    def insert(self, key):
        self.__root.insert(key)

    def delete(self, key) -> bool:
        return self.__root.delete(key)


class TrieNode:
    def __init__(self, is_end_node: bool = False):
        self.__children: Dict[str, TrieNode] = {}
        self.__is_end_node: bool = is_end_node

    def __repr__(self) -> str:
        return (
            "{"
            + ", ".join(
                [f"{k}: {v.__repr__()}" for k, v in self.__children.items()]
            )
            + "}"
        )

    def search(self, key) -> bool:
        if key == "" and self.__is_end_node:
            return True
        elif key[0] in self.__children.keys():
            return self.__children[key[0]].search(key[1:])
        return False

    def insert(self, key):
        if key == "":
            self.__is_end_node = True
            return
        elif key[0] not in self.__children.keys():
            self.__children[key[0]] = TrieNode()
        self.__children[key[0]].insert(key[1:])

    def delete(self, key) -> bool:
        if key == "" and self.__is_end_node:
            self.__is_end_node = False
            return True
        elif key[0] in self.__children.keys():
            return self.__children[key[0]].delete(key[1:])
        return False


if __name__ == "__main__":
    t = Trie()
    t.insert("hello")
    t.insert("help me")
    t.insert("world")
