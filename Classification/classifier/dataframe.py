from __future__ import annotations
from csv import reader
from collections import defaultdict
from operator import itemgetter


class DataFrame(object):
    """
    DataFrame
    ===

    [Raw Data]
      age,     income, student, credit_rating
      <=30,    low,    no,      fair
      <=30,    medium, yes,     fair
      31...40, low,    no,      fair
      >40,     high,   no,      fair
      >40,     low,    yes,     excellent

    [Data Frame]
    - columns (4):
      ['age', 'income', 'student', 'credit_rating']
    - attributes (4, R)
      [0] ['<=30', '31...40', '>40']
      [1] ['low', 'medium', 'high']
      [2] ['no', 'yes']
      [3] ['fair', 'excellent']
    - data (4, 5) vertical-way
      [0] [0, 0, 1, 2, 2]
      [1] [0, 1, 0, 2, 0]
      [2] [0, 1, 0, 0, 1]
      [3] [0, 0, 0, 0, 1]
    """
    columns: list[str]  # N
    attributes: list[list[str]]  # N x R
    data: list[list[int]]  # N x M
    N: int  # Column size
    M: int  # Row size

    def __init__(self, raw: dict[str, list[str]] = dict()):
        columns = list(raw.keys())
        attributes = []
        data = []
        for col in columns:
            attrs = list(set(raw[col]))
            attributes.append(attrs)
            data.append([attrs.index(x) for x in raw[col]])
        self._set(columns, attributes, data)

    @staticmethod
    def read_csv(file: str, delimiter="\t") -> dict[str, list[str]]:
        raw = dict()
        with open(file, "r") as f:
            keys = next(reader(f, delimiter=delimiter))
            values = [[] for _ in range(len(keys))]
            for line in reader(f, delimiter=delimiter):
                for i, val in enumerate(line):
                    values[i].append(val)
            raw = {k: v for k, v in zip(keys, values)}
        return raw

    def save_csv(self, file: str, delimiter="\t") -> None:
        with open(file, "w") as f:
            f.write(delimiter.join(self.columns) + "\n")
            for m in range(self.M):
                f.write(delimiter.join([self.attributes[n][self.data[n][m]]
                        for n in range(self.N)]) + "\n")

    def groupby(self, col: str, keep=False) -> dict[str, list[int]]:
        res = defaultdict(list[int])
        idx: int = self.columns.index(col)
        attrs: list[str] = self.attributes[idx]
        if self.M == 0:
            return dict()
        if keep:
            for attr in attrs:
                res[attr] = []
        data: list[str] = self.data[idx]
        for idx, d in enumerate(data):
            res[attrs[d]].append(idx)
        return dict(res)

    def row(self, m: int) -> list[str]:
        return [self.attributes[n][self.data[n][m]] for n in range(self.N)]

    def _set(self, cols: list[str], attrs: list[list[str]], data: list[list[int]]) -> DataFrame:
        assert isinstance(cols, list)
        self.columns = cols
        self.attributes = attrs
        self.data = data
        self.N = len(cols)
        self.M = len(data[0]) if len(self.data) > 0 else 0
        return self

    def _drow(self, m: int) -> list[list[int]]:  # (N, 1) - vertical
        return [[self.data[n][m]] for n in range(self.N)]

    def _drows(self, ms: list[int]) -> list[list[int]]:  # (N, m) - vertical
        if len(ms) == 0:
            return []
        if len(ms) == 1:
            return self._drow(ms[0])
        return [list(itemgetter(*ms)(self.data[n])) for n in range(self.N)]

    def _dcol(self, n: int) -> list[list[int]]:  # (1, M) - vertical
        if self.M == 0:
            return []
        else:
            return [self.data[n]]

    def _dcols(self, ns: list[int]) -> list[list[int]]:  # (n, M) - vertical
        if self.M == 0:
            return []
        else:
            return [self._dcol(n)[0] for n in ns]

    def _row(self, m: int) -> DataFrame:
        return DataFrame()._set(self.columns, self.attributes, self._drow(m))

    def _rows(self, ms: list[int]) -> DataFrame:
        return DataFrame()._set(self.columns, self.attributes, self._drows(ms))

    def _col(self, n: int) -> DataFrame:
        columns = [self.columns[n]]
        attributes = [self.attributes[n]]
        return DataFrame()._set(columns, attributes, self._dcol(n))

    def _cols(self, ns: list[int]) -> list[list[str]]:
        if len(ns) == 0:
            raise TypeError("Invalid argument type")
        elif len(ns) == 1:
            return self._col(ns[0])
        columns = list(itemgetter(*ns)(self.columns))
        attributes = list(itemgetter(*ns)(self.attributes))
        return DataFrame()._set(columns, attributes, self._dcols(ns))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._row(item)
        elif isinstance(item, tuple):
            m, n = item
            if isinstance(m, int) and isinstance(n, int):
                return self.attributes[n][self.data[n][m]]
            else:
                raise TypeError("Unsupported argument type")
        elif isinstance(item, list):
            if len(item) == 0 or isinstance(item[0], int):
                return self._rows(item)
            elif isinstance(item[0], str):
                return self._cols([self.columns.index(i) for i in item])
            else:
                raise TypeError("Invalid argument type")
        elif isinstance(item, str):
            return self._col(self.columns.index(item))
        raise TypeError("Invalid argument type")

    def __add__(self, other: DataFrame) -> DataFrame:
        columns = self.columns + other.columns
        attr = self.attributes + other.attributes
        data = self.data + other.data
        return DataFrame()._set(columns, attr, data)

    def __len__(self):
        return self.M

    def __repr__(self):
        lines = []
        pad = 10
        threshold = 20
        lines.append(map(lambda s: s.rjust(pad), self.columns))
        show = range(self.M) if self.M < threshold else [0, 1, -2, -1]
        for m in show:
            lines.append([self.attributes[n][self.data[n][m]].rjust(pad)
                         for n in range(self.N)])
        lines = list(map(lambda l: "\t".join(l), lines))
        if self.M != len(lines) - 1:
            lines.insert(3, "...")
        lines.insert(0, f"DataFrame ({self.M}, {self.N})")
        return "\n".join(lines)
