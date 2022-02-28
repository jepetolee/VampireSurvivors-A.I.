import torch
import RL.agent as agent
import RL.model as module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gc
import time

gamma = 0.75
weight = 0.84
tau = 0.97


def train():
    for i in range(5):
        print("시작하기까지..." + str(i + 1) + "초 남았습니다.")
        time.sleep(1)

    epoch = 100000
    model = module.A2C()
    device = torch.device('cpu')
    model.to(device)
    # print(torch.load('./save.pt'))
    model.load_state_dict(torch.load('./save.pt'))

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    while epoch > 0:

        s_list, a_list, r_list, policy_list, mcts_list = agent.run_once(model, device)
        s_len = len(s_list)
        # protection of memory-error

        num = int(s_len / 30)
        left = s_len % 30
        if left > 0:
            num += 1

        # model.to('cpu')
        for t in range(num):
            if num > t + 1:
                s_list1 = s_list[s_len - 1 - (t + 1) * 30:-1 - t * 30]
                mcts_list1 = mcts_list[s_len - 1 - (t + 1) * 30:-1 - t * 30]
                if t != 0:
                    s_prime_list = s_list[s_len - (t + 1) * 30:-t * 30]
                    mcts_prime_list = mcts_list[s_len - (t + 1) * 30:-t * 30]
                else:
                    s_prime_list = s_list[s_len - (t + 1) * 30:]
                    mcts_prime_list = mcts_list[s_len - (t + 1) * 30:]
                r_list1 = r_list[s_len - 1 - (t + 1) * 30:-1 - t * 30]
                a_list1 = a_list[s_len - 1 - (t + 1) * 30:-1 - t * 30]
                policy_list1 = policy_list[s_len - 1 - (t + 1) * 30:-1 - t * 30]
            else:
                s_list1 = s_list[1:left]
                mcts_list1 = mcts_list[1:left]
                s_prime_list = s_list[:left - 1]
                mcts_prime_list = mcts_list[:left - 1]
                r_list1 = r_list[1:left]
                a_list1 = a_list[1:left]
                policy_list1 = policy_list[1:left]
            if len(s_list1) == 0 or len(s_prime_list) == 0:
                break

            for i in range(3):
                model.set_recurrent_buffers(buf_size=len(r_list1))

                s_prime_list = np.array(s_prime_list)
                s_prime_vec = torch.tensor(s_prime_list).float().reshape(-1, 1, 1080, 1724).to('cpu')

                mcts_prime_list = np.array(mcts_prime_list)
                mcts_prime_vec = torch.tensor(mcts_prime_list).float().reshape(-1, 3000).to('cpu')

                G = model.value(s_prime_vec, mcts_prime_vec).squeeze(1).to('cpu')

                del s_prime_vec
                del mcts_prime_vec

                r_vec = torch.tensor(r_list1).float().to('cpu')
                target_vec = r_vec + gamma * G
                del r_vec
                del G
                gc.collect()

                s_list1 = np.array(s_list1)
                s_vec = torch.tensor(s_list1).float().reshape(-1, 1, 1080, 1724).to('cpu')

                mcts_list1 = np.array(mcts_list1)
                mcts_vec = torch.tensor(mcts_list1).float().reshape(-1, 3000).to('cpu')
                value_s_vec = model.value(s_vec, mcts_vec).squeeze(1).to('cpu')
                delt = (target_vec - value_s_vec).detach().to('cpu').numpy()

                policy_list1 = np.array(policy_list1)
                prior_policy = torch.from_numpy(policy_list1).reshape(-1).unsqueeze(-1).to('cpu')

                advantage_list = []
                advantage = 1e-2
                for td_error in delt[::-1]:
                    advantage = gamma * tau * advantage + td_error
                    advantage_list.append([advantage])
                advantage_list.reverse()

                advantage_vec = torch.tensor(advantage_list, dtype=torch.float).to('cpu')
                del advantage_list

                advantage_vec = (advantage_vec - advantage_vec.mean()) / (advantage_vec.std() + (2e-5))
                a_list1 = np.array(a_list1)
                a_vec = torch.tensor(a_list1, dtype=torch.float).reshape(-1).unsqueeze(-1).to('cpu')
                pi_val = model.pi(s_vec, mcts_vec, softmax_dim=1).to('cpu')

                del s_vec
                del mcts_vec

                pi_all = Categorical(pi_val)

                del pi_val

                ratio = torch.exp(pi_all.log_prob(a_vec) - prior_policy)

                del pi_all
                del prior_policy
                del a_vec

                surrogated_loss1 = ratio * advantage_vec
                surrogated_loss2 = torch.clamp(ratio, 0.9, 1.1) * advantage_vec
                loss = - torch.min(surrogated_loss1, surrogated_loss2).mean() \
                       + F.smooth_l1_loss(value_s_vec, target_vec.detach()).mean() * weight

                del surrogated_loss1
                del surrogated_loss2
                del target_vec
                del value_s_vec
                gc.collect()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        if epoch % 20 == 0:
            torch.save(model.state_dict(), "./save.pt")
        epoch -= 1

        model.del_dat()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
