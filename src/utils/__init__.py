class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        self.parent[self.find(a)] = self.find(b)


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
