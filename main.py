from sfc import SfcGenerator
from net import NetGenerator


def main():
    g = SfcGenerator()
    s = g.get_chain(0)
    print(s)
    net_g = NetGenerator()
    print(net_g.get_g())


if __name__ == "__main__":
    main()