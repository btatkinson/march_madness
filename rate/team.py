import numpy as np

from settings import *

from math import log

def log_loss(true, pred, eps=1e-15):
    p = np.clip(pred, eps, 1-eps)
    if true == 1:
        return -log(p)
    else:
        return -log(1-p)

class Team(object):
    """docstring for Team."""
    def __init__(self, tid):
        super(Team, self).__init__()
        # team id
        self.tid = tid

        self.init_ratings()
        self.init_errors()

    def init_ratings(self):
        self.gp = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.wl = 0.5

        self.elo = elo_set['initial']
        self.mov = mov_set['initial']
        self.dim = dim_set['initial']
        self.dimv = dimv_set['initial']
        self.glicko = glicko_set['initial']
        self.steph = steph_set['initial']
        return

    def init_errors(self):
        self.wlm_error = 0
        self.elo_error = 0
        self.mov_error = 0
        self.dim_error = 0
        self.dimv_error = 0
        self.glicko_error = 0
        self.steph_error = 0
        return

    def played_game(self):
        self.gp += 1
        return

    def add_win(self):
        self.wins+=1
        return

    def add_loss(self):
        self.losses+=1
        return

    def add_tie(self):
        self.ties+=1
        return

    def calc_win_loss(self):
        self.wl = (self.wins + 0.5*self.ties)/self.gp
        return

    def add_errors(self, result, error_dict):
        self.wlm_error += log_loss(result, error_dict["wlm"])
        self.elo_error += log_loss(result, error_dict["elo"])
        self.dim_error += log_loss(result, error_dict["dim"])
        self.mov_error += log_loss(result, error_dict["mov"])
        self.dimv_error += log_loss(result, error_dict["dimv"])
        # self.glicko_error += log_loss(result, error_dict["glicko"])
        # self.steph_error += log_loss(result, error_dict["steph"])

        return




# end
