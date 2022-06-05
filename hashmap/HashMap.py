from abc import ABC, abstractmethod
from hashlib import md5
from typing import Iterator, List, Optional, Tuple


class HashStrategy(ABC):
    """Superclass for all hashing strategies. Meant to be implemented and then passed to a hash map."""

    @abstractmethod
    def hash(self, key: str) -> int:
        """Hash a key into an integer value.

        Ideally the hash function is uniformly distributed over the key space.

        Args:
            key (str): Key to hash.

        Returns:
            int: Integer value of the key.
        """
        pass


class Md5HashStrategy(HashStrategy):
    def hash(self, key: str) -> int:
        """Hash a key via MD5.

        Args:
            key (str): Key to hash.

        Returns:
            int: Integer value of the key.
        """
        return int(md5(key.encode("utf-8")).hexdigest(), 16)


class HashMap:
    def __init__(
        self, strategy: HashStrategy, base_size: int = 4, max_load: float = 0.5
    ) -> None:
        self.__size = 0
        self.__strategy = strategy
        self.__max_size = base_size
        self.__max_load = 0.5

        self.__data: List[List[Tuple[str, str]]] = [
            [] for _ in range(self.__max_size)
        ]

    def __load_factor(self) -> float:
        """Get the current load factor of the hash map, which is simply the current number of elements divided by the max number of elements.

        Returns:
            float: Ratio of the current number of elements to the maximum allowed number of elements.
        """
        return (self.__size) / (self.__max_size)

    def __resize(self) -> None:
        """Resize the current hash map to twice its current size."""

        new_max_size = self.__max_size * 2
        new_data: List[List[Tuple[str, str]]] = [
            [] for _ in range(new_max_size)
        ]

        # Have to rehash everything.
        self.__max_size = new_max_size
        for k, v in self:
            new_data[self.__hash_index(k)].append((k, v))
        self.__data = new_data

    def __hash(self, key: str) -> int:
        """Use the HashStrategy to hash a key into an integer.

        Args:
            key (str): Key to hash.

        Returns:
            int: Integer value of the key.
        """
        return self.__strategy.hash(key)

    def __hash_index(self, key: str) -> int:
        """Convenience function to hash a key and return the integer value modulo the current max size.

        Args:
            key (str): The key to apply the hash function to.

        Returns:
            int: Integer value of the key modulo the current max size.
        """
        return self.__hash(key) % self.__max_size

    def put(self, key: str, value: str) -> None:
        """Insert a value into the hash map.

        Args:
            key (str): _description_
            value (str): _description_
        """
        if self.get(key) is not None:
            self.remove(key)

        self.__data[self.__hash_index(key)].append((key, value))
        self.__size += 1

        if self.__load_factor() > self.__max_load:
            self.__resize()

    def get(self, key: str) -> Optional[str]:
        """Get a value from the hash map.

        Args:
            key (str): Key to retrieve.

        Returns:
            Optional[str]: String value if found, None otherwise.
        """
        values = self.__data[self.__hash_index(key)]
        for k, v in values:
            if k == key:
                return v
        return None

    def remove(self, key: str) -> bool:
        """Remove a value from a hash map.

        Args:
            key (str): Key to retrieve.

        Returns:
            bool: True if removed, false otherwise.
        """
        values = self.__data[self.__hash_index(key)]
        for pair in values:
            if pair[0] == key:
                self.__data[self.__hash_index(key)].remove(pair)
                self.__size -= 1
                return True
        return False

    def __len__(self) -> int:
        """Get the current number of elements in the hash map."""
        return self.__size

    def __contains__(self, key: str) -> bool:
        """Does the hash map contain the given key?

        Args:
            key (str): Key to test.

        Returns:
            bool: True if found, false otherwise.
        """
        return self.get(key) is not None

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over the know key-value pairs.

        Yields:
            Iterator[Tuple[str, str]]: Iterator over each key-value pair.
        """
        for values in self.__data:
            for pair in values:
                yield pair


if __name__ == "__main__":
    hm = HashMap(Md5HashStrategy())
    hm.put("a", "1")
    hm.put("b", "2")
    hm.put("c", "3")
    hm.put("d", "4")
    hm.put("e", "5")
    hm.put("f", "6")
