from sfc import SfcGenerator
from net import NetGenerator
from solution import solve


def main():
    my_net = NetGenerator().get_g()
    g = SfcGenerator(my_net)
    t = 0
    s = g.get_chain(t)
    res = solve(my_net, s, t)
    print(res)


if __name__ == "__main__":
    main()