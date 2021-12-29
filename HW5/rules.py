class Rules:
    def __init__(self, rules: str):
        self.__gh: list[tuple[str, str, str]] = []

        self.__gv: list[tuple[str, str, str]] = []

        self.__g: list[tuple[str, str]] = []

    def __parse_rules(self, rules: str) -> None:
        raise NotImplementedError

    def create_gh(self, left_symbol: str, right_s1: str, right_s2: str) -> None:
        self.__gh.append(left_symbol, right_s1, right_s2)

    def create_gv(self, left_symbol: str, right_s1: str, right_s2: str) -> None:
        self.__gv.append(left_symbol, right_s1, right_s2)

    def create_g(self, s1: str, s2: str) -> None:
        self.__g.append(s1, s2)

    def gh(self, left_symbol: str, right_s1: str, right_s2: str) -> bool:
        return (left_symbol, right_s1, right_s2) in self.__gh

    def gv(self, left_symbol: str, right_s1: str, right_s2: str) -> bool:
        return (left_symbol, right_s1, right_s2) in self.__gv

    def g(self, s1: str, s2: str) -> bool:
        return (s1, s2) in self.__g


horizontal = [
            ('i', 'i', 'v0'),
            ('i', 'i_','v0_'),
            ('i_', 'i', 'v1'),
            ('i_', 'i_', 'v1_')
        ]

vertical = [
            ('A00', '0', '0'),
            ('A01', '0', '1'),
            ('A10', '1', '0'),
            ('A11', '1', '1'),
            ('v0','A00', '0'),
            ('v1', 'A01', '1'),
            ('v0', 'A10', '1'),
            ('v0', 'A11', '0'),
            ('v1', 'A00', '1'),
            ('v1_', 'A01', '0'),
            ('v1_', 'A10', '0'),
            ('v1_', 'A11', '1')
        ]
rename = [
            ('i', 'v0'),
            ('i_', 'v1')
        ]