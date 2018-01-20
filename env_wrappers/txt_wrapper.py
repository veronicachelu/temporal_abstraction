import scipy.ndimage
import tkinter
from PIL import Image
from PIL import ImageTk
import numpy as np
from PIL import Image
import scipy.ndimage
import numpy as np
import random
from gym import spaces

class GridWorld:
  def __init__(self, goal_locations, load_path=None):
    self.action_space = spaces.Discrete(4)

    self.rewardFunction = None
    self.nb_actions = 4
    if load_path != None:
      self.read_file(load_path)
      self.set_goal_locations(goal_locations)

    self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(self.nb_rows, self.nb_cols, 3))
    self.agentX, self.agentY = self.startX, self.startY
    self.nb_states = self.nb_rows * self.nb_cols

    self.win = tkinter.Toplevel()

    screen_width = self.win.winfo_screenwidth()
    screen_height = self.win.winfo_screenheight()

    # calculate position x and y coordinates
    x = screen_width + 100
    y = screen_height + 100
    self.h = self.MDP.shape[0] * 42
    self.w = self.MDP.shape[1] * 42
    self.win.geometry('%sx%s+%s+%s' % (self.w, self.h, x, y))
    self.win.title("Gridworld")

  def set_goal_locations(self, goal_locations):
    self.goal_locations = goal_locations
    print(self.goal_locations)

  def set_goal(self, episode_nb):
    print(episode_nb)
    print(episode_nb % 1000)
    goal_pair = self.goal_locations[episode_nb % 1000]
    self.goalX = goal_pair[0]
    self.goalY = goal_pair[1]

  def render(self, s):
    # time.sleep(0.1)
    # s = self.pix_state
    screen = scipy.misc.imresize(s, [self.h, self.w,  3], interp='nearest')
    screen = Image.fromarray(screen, 'RGB')
    # screen = screen.resize((self.w, self.h))
    # screen_width = self.win.winfo_screenwidth()
    # screen_height = self.win.winfo_screenheight()
    # x = screen_width + 100
    # y = screen_height + 100
    #
    # self.win.geometry('%sx%s+%s+%s' % (512, 512, x, y))

    tkpi = ImageTk.PhotoImage(screen)
    label_img = tkinter.Label(self.win, image=tkpi)
    label_img.place(x=0, y=0,
                    width=self.w, height=self.h)

    # self.win.mainloop()            # wait until user clicks the window
    self.win.update_idletasks()
    self.win.update()

  def build_screen(self):
    mdp_screen = np.array(self.MDP)
    mdp_screen = np.expand_dims(mdp_screen, 2)
    mdp_screen[mdp_screen == -1] = 255
    mdp_screen = np.tile(mdp_screen, [1, 1, 3])
    mdp_screen[self.agentX, self.agentY] = [0, 255, 0]
    mdp_screen[self.goalX, self.goalY] = [255, 0, 0]
    self.pix_state = mdp_screen
    self.pix_state /= 255.
    self.pix_state -= 0.5
    self.pix_state *= 2.
    # self.pix_state = scipy.misc.imresize(mdp_screen, [200, 200, 3], interp='nearest')
    return self.pix_state
    # return mdp_screen

  def reset(self):
    s = self.get_initial_state()
    screen = self.build_screen()

    return screen

  def read_file(self, load_path):
    with open(load_path, "r") as f:
      lines = f.readlines()
    self.nb_rows, self.nb_cols = lines[0].split(',')
    self.nb_rows, self.nb_cols = int(self.nb_rows), int(self.nb_cols)
    self.MDP = np.zeros((self.nb_rows, self.nb_cols))
    lines = lines[1:]
    for i in range(self.nb_rows):
      for j in range(self.nb_cols):
        if lines[i][j] == '.':
          self.MDP[i][j] = 0
        elif lines[i][j] == 'X':
          self.MDP[i][j] = -1
        elif lines[i][j] == 'S':
          self.MDP[i][j] = 0
          self.startX = i
          self.startY = j
        else:  # 'G'
          self.MDP[i][j] = 0
          self.goalX = i
          self.goalY = j

  def get_state_index(self, x, y):
    idx = y + x * self.nb_cols
    return idx

  def get_start(self):
    while True:
      startX = random.randrange(0, self.nb_rows, 1)
      startY = random.randrange(0, self.nb_cols, 1)
      if self.MDP[startX][startY] != -1 and (startX != self.goalX or startY != self.goalY):
        break

    start_inx = self.get_state_index(startX, startY)

    return start_inx, startX, startY

  def get_initial_state(self):
    agent_state_index = self.get_state_index(self.startX, self.startY)
    # agent_state_index, self.startX, self.startY = self.get_start()
    self.agentX, self.agentY = self.startX, self.startY
    return agent_state_index

  def move_goal(self):
    while True:
      goalX = random.randrange(0, self.nb_rows, 1)
      goalY = random.randrange(0, self.nb_cols, 1)
      if self.MDP[goalX][goalY] != -1 and (goalX != self.startX or goalY != self.startY):
        break

    goal_indx = self.get_state_index(goalX, goalY)
    self.goalX = goalX
    self.goalY = goalY

  def get_next_state(self, a):
    action = ["up", "right", "down", "left", 'terminate']
    nextX, nextY = self.agentX, self.agentY

    if action[a] == 'terminate':
      return -1, -1

    if self.MDP[self.agentX][self.agentY] != -1:
      if action[a] == 'up' and self.agentX > 0:
        nextX, nextY = self.agentX - 1, self.agentY
      elif action[a] == 'right' and self.agentY < self.nb_cols - 1:
        nextX, nextY = self.agentX, self.agentY + 1
      elif action[a] == 'down' and self.agentX < self.nb_rows - 1:
        nextX, nextY = self.agentX + 1, self.agentY
      elif action[a] == 'left' and self.agentY > 0:
        nextX, nextY = self.agentX, self.agentY - 1

    if self.MDP[nextX][nextY] != -1:
      return nextX, nextY
    else:
      return self.agentX, self.agentY

  def special_get_next_state(self, a, orig_nextX, orig_nextY):
    action = ["up", "right", "down", "left", 'terminate']

    nextX, nextY = orig_nextX, orig_nextY

    if action[a] == 'terminate':
      return -1, -1

    if self.MDP[orig_nextX][orig_nextY] != -1:
      if action[a] == 'up' and orig_nextY > 0:
        nextX, nextY = orig_nextX - 1, orig_nextY
      elif action[a] == 'right' and orig_nextY < self.nb_cols - 1:
        nextX, nextY = orig_nextX, orig_nextY + 1
      elif action[a] == 'down' and self.agentX < self.nb_rows - 1:
        nextX, nextY = orig_nextX + 1, orig_nextY
      elif action[a] == 'left' and orig_nextY > 0:
        nextX, nextY = orig_nextX, orig_nextY - 1

    if self.MDP[nextX][nextY] != -1:
      return nextX, nextY
    else:
      return orig_nextX, orig_nextY

  def is_terminal(self, nextX, nextY):
    if nextX == self.goalX and nextY == self.goalY:
      return True
    else:
      return False

  def get_next_reward(self, nextX, nextY):
    if self.rewardFunction is None:
      if nextX == self.goalX and nextY == self.goalY:
        reward = 1
      else:
        reward = 0
    elif len(self.rewardFunction) != self.nb_states and self.network != None and self.sess != None:
      currStateIdx = self.get_state_index(self.agentX, self.agentY)
      s, _, _ = self.get_state(currStateIdx)
      feed_dict = {self.network.observation: np.stack([s])}
      fi = self.sess.run(self.network.fi,
                    feed_dict=feed_dict)[0]
      nextStateIdx = self.get_state_index(nextX, nextY)
      s1, _, _ = self.get_state(nextStateIdx)
      feed_dict = {self.network.observation: np.stack([s1])}
      fi1 = self.sess.run(self.network.fi,
                         feed_dict=feed_dict)[0]
      reward = self.cosine_similarity((fi1 - fi), self.rewardFunction)


    else:
      currStateIdx = self.get_state_index(self.agentX, self.agentY)
      nextStateIdx = self.get_state_index(nextX, nextY)

      reward = self.rewardFunction[nextStateIdx] \
               - self.rewardFunction[currStateIdx]

    return reward

  def cosine_similarity(self, next_sf, evect):
    state_dif_norm = np.linalg.norm(next_sf)
    state_dif_normalized = next_sf / (state_dif_norm + 1e-8)
    # evect_norm = np.linalg.norm(evect)
    # evect_normalized = evect / (evect_norm + 1e-8)
    res = np.dot(state_dif_normalized, evect)
    # if  res < 0:
    #   res = -1
    # elif res > 0:
    #   res = 1
    return res

  def fake_get_state(self, idx):
    orig_agentX, orig_agentY = self.agentX, self.agentY
    x, y = self.get_state_xy(idx)
    self.agentX, self.agentY = x, y

    screen = self.build_screen()
    self.agentX, self.agentY = orig_agentX, orig_agentY

    return screen, x, y

  def get_state(self, idx):
    x, y = self.get_state_xy(idx)
    self.agentX, self.agentY = x, y

    screen = self.build_screen()

    return screen, x, y

  def not_wall(self, i, j):
    if self.MDP[i][j] != -1:
      return True
    else:
      return False

  def get_state_xy(self, idx):
    y = idx % self.nb_cols
    x = int((idx - y) / self.nb_cols)

    return x, y

  def get_next_state_and_reward(self, currState, a):
    if currState == self.nb_states:
      return currState, 0

    tmpx, tmpy = self.agentX, self.agentY
    self.agentX, self.agentY = self.get_state_xy(currState)
    nextX, nextY = self.agentX, self.agentY

    nextStateIdx = None
    reward = None

    nextX, nextY = self.get_next_state(a)
    if nextX != -1 and nextY != -1:  # If it is not the absorbing state:
      reward = self.get_next_reward(nextX, nextY)
      nextStateIdx = self.get_state_index(nextX, nextY)
    else:
      reward = 0
      nextStateIdx = self.nb_states

    self.agentX, self.agentY = tmpx, tmpy

    return nextStateIdx, reward

  def get_agent(self):
    return self.agentX, self.agentY

  def step(self, a):
    nextX, nextY = self.get_next_state(a)

    self.agentX, self.agentY = nextX, nextY

    done = False
    if self.is_terminal(nextX, nextY):
      done = True

    reward = self.get_next_reward(nextX, nextY)
    nextStateIdx = self.get_state_index(nextX, nextY)

    screen = self.build_screen()

    return screen, reward, done, nextStateIdx

  def fake_step(self, a):
    orig_agentX, orig_agentY = self.agentX, self.agentY
    nextX, nextY = self.get_next_state(a)

    self.agentX, self.agentY = nextX, nextY

    done = False
    if self.is_terminal(nextX, nextY):
      done = True

    reward = self.get_next_reward(nextX, nextY)
    nextStateIdx = self.get_state_index(nextX, nextY)

    screen = self.build_screen()

    self.agentX, self.agentY = orig_agentX, orig_agentY

    return screen, reward, done, nextStateIdx

  def special_step(self, a, last_state_idx):
    x, y = self.get_state_xy(last_state_idx)
    nextX, nextY = self.special_get_next_state(a, x, y)

    # new_x, new_y = nextX, nextY

    done = False
    if self.is_terminal(nextX, nextY):
      done = True

    reward = self.get_next_reward(nextX, nextY)
    nextStateIdx = self.get_state_index(nextX, nextY)

    screen = self.build_screen()

    return screen, reward, done, nextStateIdx

  def get_action_set(self):
    return range(0, 4)

  def define_reward_function(self, vector):
    self.rewardFunction = vector

  def define_network(self, net):
    self.network = net

  def define_session(self, sess):
    self.sess = sess

if __name__ == '__main__':

  import time
  import tkinter
  from PIL import Image
  from PIL import ImageTk
  import numpy as np
  from tools import wrappers

  player_rng = np.random.RandomState(0)
  # game = GridWorld("../mdps/longI.mdp")
  game = GridWorld("../mdps/4rooms.mdp")
  game = wrappers.LimitDuration(game, 100000)
  game = wrappers.FrameResize(game, (13,13))
  game = wrappers.ConvertTo32Bit(game)

  start = time.time()
  # reward_color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
  # reward_color = [1,0,0]
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
