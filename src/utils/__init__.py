class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.removed = set()

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def remove(self, j: int):
        """
        Detaches j from its group by making it its own root.
        All other members of the group are unaffected.
        """
        self.parent[j] = j  # j becomes its own isolated root
        self.rank[j] = 0
        self.removed.add(j)

    def connected(self, x: int, y: int) -> bool:
        if x in self.removed or y in self.removed:
            return False
        return self.find(x) == self.find(y)


def generate_square_edges(
    p: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        ((p[0] - 0.5, p[1] + 0.5), (p[0] + 0.5, p[1] + 0.5)),
        ((p[0] - 0.5, p[1] - 0.5), (p[0] + 0.5, p[1] - 0.5)),
        ((p[0] - 0.5, p[1] - 0.5), (p[0] - 0.5, p[1] + 0.5)),
        ((p[0] + 0.5, p[1] - 0.5), (p[0] + 0.5, p[1] + 0.5)),
    ]


def intersect(
    p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int], p4: tuple[int, int]
) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:
        return False
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:
        return False
    return True
