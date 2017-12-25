import csv

games = {}
with open('dqn_mean.csv', "r") as f:
  with open("dqn_std.csv", "r") as g:
    reader1 = csv.reader(f)
    reader2 = csv.reader(g)

    for i, (row1, row2) in enumerate(zip(reader1, reader2)):
      if i == 0:
        continue
      games[row1[0]] = ["{} ({})".format(m, s) for m, s in zip(row1[1:], row2[1:])]

for key, val in games.items():
    print(" & ".join([key] + val) + " \\\\")
# print(games)