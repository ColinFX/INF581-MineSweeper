from display import *


seeds = 20
times = 10
method = 2
ratio = 1.4
cut = 10

note = {}
wins = {i: 0 for i in range(seeds)}
loss = {i: 0 for i in range(seeds)}
for i in range(seeds):
    for j in range(times):
        d = Display(i, method, ratio, cut)
        note[i] = d.mainLoop()
        if note[i][-1]["state"] == "gamewin":
            wins[i] += 1
        else:
            loss[i] += 1


from render import *

render = render()


def rend(note):
    for s in note.keys():
        for i in range(len(note[s])):
            render.gameStageDrawAuto(note[s][i], s, i, method)


# rend(note)
# print(wins)
# print(loss)
print(ratio, end=' ')
print(sum(wins.values()) / (sum(wins.values()) + sum(loss.values())))
