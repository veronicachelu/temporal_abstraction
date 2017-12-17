import numpy as np

class RingBuffer():

  def __init__(self, shape):
    self.buffer_size = shape[0]
    self.data = np.zeros(shape, dtype='f')
    self.index = -1
    self.effective_size = 0
    self.full = False

  def append(self, x):
    self.index += 1
    x_index = self.index % self.buffer_size
    self.data[x_index] = x
    self.effective_size += 1
    self.full = True if self.effective_size >= self.buffer_size else False


  def get(self):
    "Returns the first-in-first-out data in the ring buffer"
    idx = (self.index + np.arange(self.buffer_size)) % self.buffer_size
    return self.data[idx]

if __name__ == '__main__':
  ringbuff = RingBuffer((3, ))
  for i in range(1, 40):
    ringbuff.append(i)  # write
    ringbuff.get()  # read





