import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import time
import tkinter
from PIL import Image
from PIL import ImageTk
import numpy as np
from PIL import Image
from gym import spaces

class gameOb():
    def __init__(self, coordinates, size, color, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name

class Gridworld_NonMatching():
  def __init__(self, partial=False, size=5, nb_apples=1, nb_oranges=1, orange_reward=0, seed=42, deterministic=True,
               internal_render=False):
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(size, size, 3))
    self.sizeX = size
    self.sizeY = size
    self.actions = 4
    self.max_apples = self.sizeX - 1
    self.max_oranges = self.sizeX - 1
    self.nb_apples = nb_apples
    self.nb_oranges = nb_oranges
    self.deterministic = deterministic

    self.objects = []
    self.orange_reward = orange_reward
    self.partial = partial
    self.bg = np.zeros([size, size])
    self.seed = seed
    self.first_room = True

    # if internal_render:
    self.win = tkinter.Toplevel()

    screen_width = self.win.winfo_screenwidth()
    screen_height = self.win.winfo_screenheight()

    # calculate position x and y coordinates
    x = screen_width + 100
    y = screen_height + 100

    self.win.geometry('+%d+%d' % (200, 200))
    self.win.title("Gridworld")
    # self.win.bind("<Button>", button_click_exit_mainloop)
    self.old_screen_label = None

    if seed:
      np.random.seed(self.seed)
    a = self.reset()
    # plt.imshow(a_big, interpolation="nearest")

  def get_screen(self):
    state, state_big = self.renderEnv()
    return state_big

  def set_seed(self, seed):
    self.seed = seed

  def getFeatures(self):
    return np.array([self.objects[0].x, self.objects[0].y]) / float(self.sizeX)

  def reset(self):
    if self.deterministic:
      apple_color = [0, 1, 0]
    else:
      while True:
        apple_color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
        if apple_color != [0, 0, 1] and apple_color[0] != [1, 1, 0] and apple_color != [1, 1, 1] and apple_color != [0,
                                                                                                                     0,
                                                                                                                     0]:
          break
    self.choice_first_room = np.random.choice(2, 1)
    self.objects = []
    self.apple_color = apple_color
    self.orange_color = [1 - a for a in self.apple_color]
    self.orientation = 0
    self.hero = gameOb(self.newPosition(0), 1, [0, 0, 1], None, 'hero')
    self.objects.append(self.hero)
    # for i in range(self.nb_apples):
    #     apple = gameOb(self.newPosition(0), 1, self.apple_color, 1, 'apple')
    #     self.objects.append(apple)
    # for i in range(self.nb_oranges):
    #     orange = gameOb(self.newPosition(0), 1, self.orange_color, self.orange_reward, 'orange')
    #     self.objects.append(orange)
    self.teleporter = gameOb(self.newPosition(0), 1, [1, 1, 1], 1, 'teleporter')
    self.objects.append(self.teleporter)
    if self.choice_first_room[0]:
      obj = gameOb(self.newPosition(0), 1, self.apple_color, 0, 'apple')
    else:
      obj = gameOb(self.newPosition(0), 1, self.orange_color, 0, 'orange')
    self.objects.append(obj)
    state, s_big = self.renderEnv()
    self.state = state

    # for ob in self.objects:
    #     if ob.name == 'apple':
    #         zagoal = ob
    #         break

    # return state, None, None, {"goal": (zagoal.y, zagoal.x), "hero": (self.hero.y, self.hero.x), "grid": (self.sizeY, self.sizeX)}
    self.first_room = True
    return state * 255

  def moveChar(self, action):
    # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
    hero = self.objects[0]
    blockPositions = [[-1, -1]]
    for ob in self.objects:
      if ob.name == 'block': blockPositions.append([ob.x, ob.y])
    blockPositions = np.array(blockPositions)
    heroX = hero.x
    heroY = hero.y
    penalize = 0.
    if action < 4:
      if self.orientation == 0:
        direction = action
      if self.orientation == 1:
        if action == 0:
          direction = 1
        elif action == 1:
          direction = 0
        elif action == 2:
          direction = 3
        elif action == 3:
          direction = 2
      if self.orientation == 2:
        if action == 0:
          direction = 3
        elif action == 1:
          direction = 2
        elif action == 2:
          direction = 0
        elif action == 3:
          direction = 1
      if self.orientation == 3:
        if action == 0:
          direction = 2
        elif action == 1:
          direction = 3
        elif action == 2:
          direction = 1
        elif action == 3:
          direction = 0

      if direction == 0 and hero.y >= 1 and [hero.x, hero.y - 1] not in blockPositions.tolist():
        hero.y -= 1
      if direction == 1 and hero.y <= self.sizeY - 2 and [hero.x, hero.y + 1] not in blockPositions.tolist():
        hero.y += 1
      if direction == 2 and hero.x >= 1 and [hero.x - 1, hero.y] not in blockPositions.tolist():
        hero.x -= 1
      if direction == 3 and hero.x <= self.sizeX - 2 and [hero.x + 1, hero.y] not in blockPositions.tolist():
        hero.x += 1
    if hero.x == heroX and hero.y == heroY:
      penalize = 0.0
    self.objects[0] = hero
    return penalize

  def newPosition(self, sparcity):
    iterables = [list(range(self.sizeX)), list(range(self.sizeY))]
    points = []
    for t in itertools.product(*iterables):
      points.append(t)
    for objectA in self.objects:
      if (objectA.x, objectA.y) in points: points.remove((objectA.x, objectA.y))
    location = np.random.choice(list(range(len(points))), replace=False)
    return points[location]

  def checkGoal(self):
    hero = self.objects[0]
    fruits = self.objects[1:]
    for fruit in fruits:
      if hero.x == fruit.x and hero.y == fruit.y and hero != fruit:
        if self.first_room:
          if fruit.reward == 1:
            self.objects = self.objects[:1]
            for i in range(self.nb_apples):
              if self.choice_first_room[0]:
                reward = -10
              else:
                reward = 10

            apple = gameOb(self.newPosition(0), 1, self.apple_color, reward, 'apple')
            self.objects.append(apple)
            orange = gameOb(self.newPosition(0), 1, self.orange_color, -reward, 'orange')
            self.objects.append(orange)
            self.first_room = False
            return 1, False
          else:
            self.objects.remove(fruit)
            if self.choice_first_room:
              obj = gameOb(self.newPosition(0), 1, self.apple_color, 0, 'apple')
            else:
              obj = gameOb(self.newPosition(0), 1, self.orange_color, 0, 'orange')
            self.objects.append(obj)

            return 0, False
        else:
          self.objects.remove(fruit)
          return fruit.reward, True
    return 0.0, False

  def render(self):

    time.sleep(0.1)

    state, state_big = self.renderEnv()
    #
    # screen = Image.fromarray(state_big, 'RGB')
    # screen = screen.resize((512, 512))
    #
    # self.win.geometry('%dx%d' % (screen.size[0], screen.size[1]))
    #
    # tkpi = ImageTk.PhotoImage(screen)
    # label_img = tkinter.Label(self.win, image=tkpi)
    # label_img.place(x=0, y=0,
    #                 width=screen.size[0], height=screen.size[1])
    #
    # # self.win.mainloop()            # wait until user clicks the window
    # self.win.update_idletasks()
    # self.win.update()

    # import matplotlib.pyplot as plt
    # pil_image = Image.fromarray(state_big)
    # pil_image.show()
    # plt.imshow(state_big)

  def renderEnv(self):
    if self.partial == True:
      padding = 2
      a = np.ones([self.sizeY + (padding * 2), self.sizeX + (padding * 2), 3])
      a[padding:-padding, padding:-padding, :] = 0
      a[padding:-padding, padding:-padding, :] += np.dstack([self.bg, self.bg, self.bg])
    else:
      a = np.zeros([self.sizeY, self.sizeX, 3])
      padding = 0
      a += np.dstack([self.bg, self.bg, self.bg])
    try:
      hero = self.objects[0]
    except:
      print("fsf")
    for item in self.objects:
      a[item.y + padding:item.y + item.size + padding, item.x + padding:item.x + item.size + padding,
      :] = item.color
      # if item.name == 'hero':
      #    hero = item
    if self.partial == True:
      a = a[(hero.y):(hero.y + (padding * 2) + hero.size), (hero.x):(hero.x + (padding * 2) + hero.size), :]
    a_big = scipy.misc.imresize(a, [200, 200, 3], interp='nearest')
    return np.asarray(a, dtype=np.uint8), a_big

  def step(self, action):
    penalty = self.moveChar(action)
    reward, done = self.checkGoal()
    state, s_big = self.renderEnv()

    # for ob in self.objects:
    #     if ob.name == 'apple':
    #         zagoal = ob
    #         break
    zagoal = None
    if self.first_room:
      zagoal = self.teleporter
    else:
      for ob in self.objects:
        if self.choice_first_room[0] and ob.name == 'orange':
          zagoal = ob
          break
        elif not self.choice_first_room[0] and ob.name == 'apple':
          zagoal = ob
          break

    if reward == None:
      print(done)
      print(reward)
      print(penalty)
      if done:
        return state*255, (reward + penalty), done, {}
      return state*255, (reward + penalty), done, {"goal": (zagoal.y, zagoal.x), "hero": (self.hero.y, self.hero.x),
                                               "grid": (self.sizeY, self.sizeX)}
    else:

      if done:
        return state*255, (reward + penalty), done, {}
      # return state, s_big, (reward + penalty), done, [self.objects[0].y, self.objects[0].x] + [goal.y, goal.x]
      return state*255, (reward + penalty), done, {"goal": (zagoal.y, zagoal.x), "hero": (self.hero.y, self.hero.x),
                                                      "grid": (self.sizeY, self.sizeX)}
