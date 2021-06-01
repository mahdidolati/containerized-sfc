from sfc import SfcGenerator


def main():
    g = SfcGenerator()
    s = g.get_chain(0)
    print(s)


if __name__ == "__main__":
    main()