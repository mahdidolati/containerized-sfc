class TestResult:
    def __init__(self):
        self.SU = "Success"
        self.SF = "S-Failure"
        self.RF = "R-Failure"
        self.res_groups = {self.SU: 0, self.SF: 0, self.RF: 0}
        self.avg_admit = 0
        self.run_avg_admit = list()
        self.avg_dl = 0
        self.run_avg_dl = list()
        self.chain_bw = 0
        self.revenue = 0
        self.avg_dl_per_acc = 0