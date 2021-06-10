from sfc import SfcGenerator
from net import NetGenerator
from solution import solve


def main():
    my_net = NetGenerator().get_g()
    sfc_gen = SfcGenerator(my_net)
    rate = 0.0
    sampling_rate = 0.7
    n = 300
    for t in range(n):
        s = sfc_gen.get_chain(t*3)
        if solve(my_net, s, t*3, sampling_rate):
            rate = rate + 1
    print(rate / n)


if __name__ == "__main__":
    main()