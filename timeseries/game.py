import numpy as np

from elo import Elo

h_adv = 107
relo_h_adv = 45

class Game(object):
    """docstring for Game."""
    def __init__(self, team1, team2, loc, margin, reb_margin, games_played):
        super(Game, self).__init__()
        self.team1 = team1
        self.team2 = team2
        self.loc = loc
        self.margin = margin
        self.reb_margin = reb_margin
        self.games_played = games_played

    def play_game(self):

        elo = Elo()

        elo_diff = self.team1.elo - self.team2.elo
        relo_diff = self.team1.relo - self.team2.relo

        # baseline: 1.215, 1.187, 1.189, 1.1646
        if self.loc == "H":
            elo_diff += h_adv
            relo_diff += relo_h_adv
        elif self.loc == "A":
            elo_diff += h_adv
            relo_diff -= relo_h_adv

        elox = elo.get_expected(elo_diff)
        relox = elo.get_expected(relo_diff)

        self.team1.played_game()
        self.team2.played_game()

        self.team1.add_win()
        self.team2.add_loss()

        if self.reb_margin > 0:
            reb_result = 1
        elif self.reb_margin < 0:
            reb_result = 0
        elif self.reb_margin == 0:
            reb_result = 0.5

        self.team1.add_errors(1, reb_result, elox, relox)
        self.team2.add_errors(0, reb_result, (1-elox), 1-relox)

        # update ratings

        # else:
        #     dimv_K = 170.59 * (self.games_played ** -0.673)

        if self.games_played <= 4:
            elo_K = 60
        elif self.games_played <= 8:
            elo_K = 45
        elif self.games_played <= 12:
            elo_K = 31
        elif self.games_played <= 16:
            elo_K = 28.5
        elif self.games_played <= 20:
            elo_K = 22
        elif self.games_played <= 24:
            elo_K = 18
        elif self.games_played <= 28:
            elo_K = 18
        elif self.games_played <= 32:
            elo_K = 18
        else:
            elo_K = 18

        if self.games_played <= 4:
            relo_K = 60
        elif self.games_played <= 8:
            relo_K = 45
        elif self.games_played <= 12:
            relo_K = 31
        elif self.games_played <= 16:
            relo_K = 28.5
        elif self.games_played <= 20:
            relo_K = 22
        elif self.games_played <= 24:
            relo_K = 18
        elif self.games_played <= 28:
            relo_K = 18
        elif self.games_played <= 32:
            relo_K = 18
        else:
            relo_K = 18

        elo_delta = elo.get_mov_delta(1, elox, self.margin, (self.team1.elo-self.team2.elo), elo_K)
        relo_delta = elo.get_mov_delta(1, relox, self.reb_margin, (self.team1.relo-self.team2.relo), relo_K)

        self.team1.elo += elo_delta
        self.team1.relo += relo_delta
        self.team2.elo -= elo_delta
        self.team2.relo -= relo_delta

        return self.team1, self.team2








# end
