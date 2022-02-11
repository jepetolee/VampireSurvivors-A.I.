import torch

import RL.agent as agent
import RL.model as module
import pyautogui
import Capture
import torch.optim as optim
import numpy as np

gamma = 0.75


def run():
    count = 10000
    model = module.A2C()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    model.load_state_dict(torch.load('./save.pt'))
    r_latest = 0
    while count > 0:

        s_list, a_list, r_list = agent.run_once(model,r_latest)
        if r_latest < r_list[-1]:
            r_latest = r_list[-1]
        s_latest = torch.tensor(s_list[-1]).float().reshape(-1, 1, 1080, 1724)
        s_len = len(s_list)
        if s_len >50:
            s_list = s_list[s_len-50:]
            r_list = r_list[s_len - 50:]
            a_list = a_list[s_len - 50:]
        v = model.value(s_latest).detach().clone().numpy()
        G = v.reshape(-1)
        target = list()
        for r in r_list:
            G = r + gamma * G
            target.append(G)
        target= np.array(target)
        target_vec = torch.from_numpy(target).float()

        s_list = np.array(s_list)
        s_vec = torch.tensor(s_list).float().reshape(-1, 1, 1080, 1724)
        a_list = np.array(a_list)
        a_vec = torch.tensor(a_list).reshape(-1).unsqueeze(-1)
        advantage = target_vec - model.value(s_vec).reshape(-1)

        pi_val = model.pi(s_vec, softmax_dim=1)
        pi_all = pi_val.gather(1, a_vec).reshape(-1)

        loss = -(torch.log(pi_all) * advantage.detach().mean()) + \
               torch.nn.functional.l1_loss(model.value(s_vec).reshape(-1), target_vec.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        count -= 1
        if count%1==0:
            torch.save(model.state_dict(),"./save.pt")


if __name__ == "__main__":
    run()
