import  numpy as np
import torch
import Monte_Carlo_tree as mcts

mcts_worker = mcts.MCTS()
mcts_worker.backup()
mcts_worker.input([0],rollout=True)

mcts_worker.input([0],rollout=True)

mcts_worker.input([1],rollout=True)

mcts_worker.append_reward(0)
mcts_worker.save()

tensor =  np.load('./Monte_Carlo_tree/mcts.npy')

for episode in range(10):
    print(tensor[episode])
tesor2 = torch.load('./save.pt')
print(tesor2)