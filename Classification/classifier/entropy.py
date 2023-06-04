import sys, math

def H(terms: list[int]):
    s = sum(terms)
    e = sys.float_info.epsilon
    res = sum(map(lambda x: - x/(s + e) * math.log2(x/(s + e) + e), terms))
    return res if res > e else 0
