from sfc import SfcGenerator
from net import NetGenerator


def main():
    g = SfcGenerator()
    s = g.get_chain(0)
    print(s)
    my_net = NetGenerator().get_g()
    print(my_net.g.edges())
    print(my_net.get_biggest_path("e0", "c"))


if __name__ == "__main__":
    main()