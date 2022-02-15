import torch

import RL.agent as agent
import RL.model as module
import torch.optim as optim
import numpy as np

gamma = 0.75

import gc



def run():
    count = 10000
    model = module.A2C()
    device = torch.device('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
  #  model.load_state_dict(torch.load('./save.pt'))
    model.to(device)
    r_latest =0
    while count > 0:

        s_list, a_list, r_list = agent.run_once(model,r_latest)
        s_latest = torch.tensor(s_list[-1]).float().reshape(-1, 1, 1080, 1724).to(device)
        s_len = len(s_list)
        if s_len >30:
            s_list = s_list[s_len-30:-1]
            r_list = r_list[s_len - 30:-1]
            a_list = a_list[s_len - 30:-1]
        if r_latest<r_list[-1]*10:
            r_latest =r_list[-1]*10
        v = model.value(s_latest).detach().clone().to('cpu').numpy()
        G = v.reshape(-1)
        target = list()
        for r in r_list:
            G = r + gamma * G
            target.append(G)
        target= np.array(target)
        target_vec = torch.from_numpy(target).float().to(device)
        model.set_recurrent_buffers(buf_size=len(r_list))
        s_list = np.array(s_list)
        s_vec = torch.tensor(s_list).float().reshape(-1, 1, 1080, 1724).to(device)
        a_list = np.array(a_list)
        a_vec = torch.tensor(a_list).reshape(-1).unsqueeze(-1).to(device)
        advantage = target_vec - model.value(s_vec).reshape(-1).to(device)

        pi_val = model.pi(s_vec, softmax_dim=1).to(device)
        pi_all = pi_val.gather(1, a_vec).reshape(-1).to(device)

        loss = -(torch.log(pi_all) * advantage.detach()).mean() + \
               torch.nn.functional.smooth_l1_loss(model.value(s_vec).reshape(-1), target_vec.reshape(-1))
        del s_vec
        del a_vec
        del advantage
        del s_list
        del s_latest
        del target
        del target_vec
        del a_list
        del pi_val
        del pi_all

        torch.cuda.empty_cache()
        gc.collect()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        count -= 1
        torch.save(model.state_dict(),"./save.pt")
        model.del_dat()
        del loss
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run()
