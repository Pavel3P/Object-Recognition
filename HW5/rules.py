class Rules:
    def __init__(self, rules: str):
        self.__gh: list[tuple[str, str, str]] = []
        self.__gv: list[tuple[str, str, str]] = []
        self.__g: list[tuple[str, str]] = []

    def __parse_rules(self, rules: str) -> None:
        raise NotImplementedError

    def __create_gh(self, left_symbol: str, right_s1: str, right_s2: str) -> None:
        raise NotImplementedError

    def __create_gv(self, left_symbol: str, right_s1: str, right_s2: str) -> None:
        raise NotImplementedError

    def __create_g(self, s1: str, s2: str) -> None:
        raise NotImplementedError

    def gh(self, left_symbol: str, right_s1: str, right_s2: str) -> bool:
        return (left_symbol, right_s1, right_s2) in self.__gh

    def gv(self, left_symbol: str, right_s1: str, right_s2: str) -> bool:
        return (left_symbol, right_s1, right_s2) in self.__gv

    def g(self, s1: str, s2: str) -> bool:
        return (s1, s2) in self.__g
