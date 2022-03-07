import  numpy as np
import torch
import Monte_Carlo_tree as mcts

mcts_worker = mcts.MCTS()
mcts_worker.backup()
mcts_worker.input([2])

mcts_worker.input([22])

mcts_worker.input([29])

mcts_worker.append_reward(0)
mcts_worker.save()

tensor =  np.load('./Monte_Carlo_tree/mcts.npy')

for episode in range(10):
    print(tensor[episode])
'''
tesor2 = torch.load('./save.pt')
print(tesor2)
'''
