import torch

import RL.agent as agent
import RL.model as module
import pyautogui
import Capture
import torch.optim as optim
import numpy as np

gamma = 0.75


def run():
    count = 30
    model = module.A2C()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # model.load_state_dict(torch.load('./save.pt'))
    while count > 0:
        s_list, a_list, r_list = agent.run_once(model)
        s_latest = torch.tensor(s_list[-1]).float().reshape(-1, 1, 28, 28)
        v = model.value(s_latest).detach().clone().numpy()
        G = v.reshape(-1)
        target = list()
        for r in r_list:
            G = r + gamma * G
            target.append(G)
        target_vec = torch.tensor(target).float()
        s_vec = torch.tensor(s_list).float().reshape(-1, 1, 28, 28)
        a_vec = torch.tensor(a_list).reshape(-1).unsqueeze(-1)
        advantage = target_vec - model.value(s_vec).reshape(-1)

        pi_val = model.pi(s_vec, softmax_dim=1)
        pi_all = pi_val.gather(1, a_vec).reshape(-1)

        loss = -(torch.log(pi_all) * advantage.detach().mean()) + \
               torch.nn.functional.l1_loss(model.value(s_vec).reshape(-1), target_vec.reshape(-1))
        loss = torch.log(loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count -= 1
        if count%10==0:
            torch.save(model.state_dict(),"./save.pt")


if __name__ == "__main__":
    run()
