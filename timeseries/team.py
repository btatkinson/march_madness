import numpy as np

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

        self.elo = 1500
        self.relo = 1500
        return

    def init_errors(self):
        self.elo_error = 0
        self.relo_error = 0
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

    def add_errors(self, result, reb_result, elo_e, relo_e):
        self.elo_error += log_loss(result, elo_e)
        self.relo_error += log_loss(reb_result, relo_e)
        return




# end
