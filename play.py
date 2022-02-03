import RL.agent as agent
import RL.model as module
import pyautogui
import Capture
import torch.optim as optim
def run():
    model = module.A2C()
    optimizer = optim.Adam(model.parameters(),lr= 0.1)
    reward =agent.run_once(model)



if __name__ == "__main__":
    run()
