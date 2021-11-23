class Const:
    VNF_NUM = 15
    VNF_CPU = [0.1, 0.4]
    VNF_RAM = [0.1, 0.4]
    VNF_LAYER = [5, 12]  # [5, 12]
    TAU1 = [3, 8]
    TAU2 = [10, 20]
    LAYER_NUM = 20
    LAYER_SIZE = [50, 350]  # [2, 70]
    SFC_LEN = [1, 7]
    SFC_DELAY = [2000, 3000]
    ALPHA_RANGE = [0.8, 1.05]
    LAMBDA_RANGE = [1, 5]
    #
    WIRE_LINK_PR = 0.1
    LINK_BW = [100, 200]
    SERVER_CPU = [16, 32]
    SERVER_RAM = [4, 32]
    SERVER_DISK = [70, 100]


class ConstSingleQ:
    Const.LAYER_NUM = 6
    Const.VNF_LAYER = [2, 6]
    Const.VNF_NUM = 20
    Const.SFC_LEN = [1, 10]
    Const.TAU1 = [2, 7]
    Const.TAU2 = [2, 7]
    Const.LAYER_SIZE = [10, 21]
    Const.SFC_DELAY = [100, 200]
    Const.SERVER_DISK = [70, 100]
    Const.SERVER_CPU = [50, 100]
    Const.SERVER_RAM = [50, 100]
