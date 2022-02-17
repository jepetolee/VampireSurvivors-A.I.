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
    device = torch.device('cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # model.load_state_dict(torch.load('./save.pt'))
    #model.to(device)
    r_latest = 0
    while count > 0:
        s_list, a_list, r_list,policy_list = agent.run_once(model, r_latest)
        s_len = len(s_list)
        # protection of memory-error
        if s_len > 50:
            s_list1 = s_list[s_len - 51:-2]
            s_prime_list = s_list[s_len - 50:-1]
            r_list = r_list[s_len - 51:-2]
            a_list = a_list[s_len - 51:-2]
            policy_list = policy_list[s_len - 50:-1]
            s_list= s_list1
            del s_list1

        else:
            s_list1 = s_list[s_len - 11:-2]
            s_prime_list = s_list[s_len - 10:-1]
            r_list = r_list[s_len - 11:-2]
            a_list = a_list[s_len - 11:-2]
            policy_list = policy_list[s_len - 10:-1]
            s_list =s_list1
            del s_list1


        if r_latest < r_list[-1] * 10:
            r_latest = r_list[-1] * 10

        for i in range(3):
            model.set_recurrent_buffers(buf_size=len(r_list))
            s_prime_list = np.array(s_prime_list)
            s_prime_vec = torch.tensor(s_prime_list).float().reshape(-1, 1, 1080, 1724).to(device)
            G = model.value(s_prime_vec).squeeze(1).to(device)
            del s_prime_vec
            r_vec = torch.tensor(r_list).float().to(device)
            target_vec = r_vec + gamma * G
            del r_vec
            s_list = np.array(s_list)
            s_vec = torch.tensor(s_list).float().reshape(-1, 1, 1080, 1724).to(device)
            value_s_vec = model.value(s_vec).squeeze(1).to(device)
            delt = (target_vec - value_s_vec).detach().numpy()
            policy_list = np.array(policy_list)
            prior_policy = torch.from_numpy(policy_list).to(device)
            advantage_list = []
            advantage = 0.0
            for td_error in delt[::-1]:
                advantage = gamma * 0.97 * advantage + td_error
                advantage_list.append([advantage])
            advantage_vec = torch.tensor(advantage_list, dtype=torch.float).to(device)
            del advantage_list
            advantage_vec = (advantage_vec - advantage_vec.mean()) / advantage_vec.std()
            a_list = np.array(a_list)
            a_vec = torch.tensor(a_list, dtype=torch.float).reshape(-1).unsqueeze(-1).to(device)
            pi_val = model.pi(s_vec, softmax_dim=1).to(device)
            del s_vec
            print(prior_policy)
            pi_all = pi_val.squeeze(1).gather(1,prior_policy).to(device)
            del prior_policy
            del pi_val

            ratio = torch.exp(torch.log(pi_all) - torch.log(a_vec))
            del pi_all
            del a_vec
            surrogated_loss1 = ratio * advantage_vec
            surrogated_loss2 = torch.clamp(ratio, 0.9, 1.1)
            loss = - torch.min(surrogated_loss1, surrogated_loss2) + torch.nn.functional.smooth_l1_loss(value_s_vec,target_vec.detach())
            #torch.nan_to_num(loss,0)
            del surrogated_loss1
            del surrogated_loss2
            del target_vec
            del value_s_vec
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        count -= 1
        if count % 10 == 0:
            torch.save(model.state_dict(), "./save.pt")
        model.del_dat()
        del loss
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run()
