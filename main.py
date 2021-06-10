from sfc import SfcGenerator
from net import NetGenerator
from solution import solve


def main():
    my_net = NetGenerator().get_g()
    sfc_gen = SfcGenerator(my_net)
    for t in range(10):
        s = sfc_gen.get_chain(t)
        res = solve(my_net, s, t)
        print(res)


if __name__ == "__main__":
    main()