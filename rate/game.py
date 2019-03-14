import numpy as np

from settings import *
from elo import Elo

h_adv = elo_set['home']

class Game(object):
    """docstring for Game."""
    def __init__(self, team1, team2, loc, margin, games_played):
        super(Game, self).__init__()
        self.team1 = team1
        self.team2 = team2
        self.loc = loc
        self.margin = margin
        self.games_played = games_played

    def play_game(self):

        elo = Elo()

        # get expected results for all systems
        wl1 = self.team1.wl
        wl2 = self.team2.wl

        wlmx = (wl1 + (1-wl2))/2

        elo_diff = self.team1.elo - self.team2.elo
        dim_diff = self.team1.dim - self.team2.dim
        mov_diff = self.team1.mov - self.team2.mov
        dimv_diff = self.team1.dimv - self.team2.dimv

        # baseline: 1.215, 1.187, 1.189, 1.1646
        if self.loc == "H":
            elo_diff += h_adv
            dim_diff += h_adv
            mov_diff += h_adv
            dimv_diff += h_adv
        elif self.loc == "A":
            elo_diff -= h_adv
            dim_diff -= h_adv
            mov_diff -= h_adv
            dimv_diff -= h_adv

        # basic elo systems
        elox = elo.get_expected(elo_diff)
        dimx = elo.get_expected(dim_diff)

        # mov systems
        movx = elo.get_expected(mov_diff)
        dimvx = elo.get_expected(dimv_diff)

        # glicko
        # glickox = glicko.get_expected(self.team1.glicko, self.team2.glicko)

        # step
        # stephx = steph.get_expected(self.team1.steph, self.team2.steph)

        self.team1.played_game()
        self.team2.played_game()

        self.team1.add_win()
        self.team2.add_loss()

        self.team1.calc_win_loss()
        self.team2.calc_win_loss()

        p1_x = {
            "wlm":wlmx,
            "elo":elox,
            "dim":dimx,
            "mov":movx,
            "dimv":dimvx,
            # "glicko":glickox,
            # "steph":stephx
        }
        p2_x = {
            "wlm":1-wlmx,
            "elo":1-elox,
            "dim":1-dimx,
            "mov":1-movx,
            "dimv":1-dimvx,
            # "glicko":glickox,
            # "steph":stephx
        }

        self.team1.add_errors(1, p1_x)
        self.team2.add_errors(0, p2_x)

        # update ratings
        elo_K = elo_set['K']
        elo_delta = elo.get_delta(1, elox, elo_K)

        if self.games_played <= 2:
            dim_K = 170
        elif self.games_played <=4:
            dim_K = 97.5
        elif self.games_played <=5:
            dim_K = 67
        elif self.games_played <=8:
            dim_K = 67
        elif self.games_played <=10:
            dim_K = 30
        else:
            dim_K = dim_set['K']
        dim_delta = elo.get_delta(1, dimx, dim_K)

        mov_K = mov_set['K']
        mov_delta = elo.get_mov_delta(1, movx, self.margin, (self.team1.mov-self.team2.mov), mov_K)

        if self.games_played <= 4:
            dimv_K = 60
        # else:
        #     dimv_K = 170.59 * (self.games_played ** -0.673)
        elif self.games_played <= 8:
            dimv_K = 45
        elif self.games_played <= 12:
            dimv_K = 31
        elif self.games_played <= 16:
            dimv_K = 28.5
        elif self.games_played <= 20:
            dimv_K = 22
        elif self.games_played <= 24:
            dimv_K = 18
        elif self.games_played <= 28:
            dimv_K = 18
        elif self.games_played <= 32:
            dimv_K = 18
        else:
            dimv_K = dimv_set['K']
        dimv_delta = elo.get_mov_delta(1, dimvx, self.margin, (self.team1.dimv-self.team2.dimv), dimv_K)

        self.team1.elo += elo_delta
        self.team1.dim += dim_delta
        self.team1.mov += mov_delta
        self.team1.dimv += dimv_delta

        self.team2.elo -= elo_delta
        self.team2.dim -= dim_delta
        self.team2.mov -= mov_delta
        self.team2.dimv -= dimv_delta

        return self.team1, self.team2








# end
