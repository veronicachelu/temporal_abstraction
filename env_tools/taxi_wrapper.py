import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from gym.envs.registration import registry, register, make, spec

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        nS = 500
        nR = 5
        nC = 5
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0:
                                newrow = min(row+1, maxR)
                            elif a==1:
                                newrow = max(row-1, 0)
                            if a==2 and self.desc[1+row,2*col+2]==b":":
                                newcol = min(col+1, maxC)
                            elif a==3 and self.desc[1+row,2*col]==b":":
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < 4 and taxiloc == locs[passidx]):
                                    newpassidx = 4
                                else:
                                    reward = -10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==4:
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx==4:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (5) 5, 5, 4
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile


# register(
#     id='Taxi-v2',
#     entry_point='gym.envs.toy_text.taxi:TaxiEnv',
#     reward_threshold=8, # optimum = 8.46
#     max_episode_steps=200,
# )

if __name__ == '__main__':

  import time
  import tkinter
  from PIL import Image
  from PIL import ImageTk
  import numpy as np
  from env_tools import env_wrappers

  player_rng = np.random.RandomState(0)
  game = TaxiEnv()
  game = env_wrappers.LimitDuration(game, 200)
  # game = wrappers.FrameResize(game, (13,13))
  # game = wrappers.ConvertTo32Bit(game)

  start = time.time()
  s = game.reset()
  ep = 0
  step = 0
  tot_rw = 0
  ep_r = 0

  while True:
    s, r, d, _ = game.step(player_rng.choice(4))
    step += 1
    ep_r += r
    game.render(s)
    tot_rw += r
    if d:
      ep += 1
      print("ep {} reward is {} ep steps {}".format(ep, ep_r, step))
      ep_r = 0
      step = 0
      s = game.reset()

  print("Finished %d episodes in %d steps in %.2f. Total reward: %d.",
        (ep, step, time.time() - start, tot_rw))
