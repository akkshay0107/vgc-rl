import torch
import torch.nn as nn
from torch.distributions import Categorical
from encoder import ACT_SIZE


class PseudoPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=128):
        super().__init__()
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs_tensor: torch.Tensor, action_mask: torch.Tensor):
        batch_size = obs_tensor.shape[0]
        obs_flat = obs_tensor.view(batch_size, -1).to(self.device)
        logits = self.network(obs_flat)
        logits = logits.view(batch_size, 2, self.action_dim)

        # mask fill logits output of network
        mask = action_mask == 0
        logits = logits.masked_fill(mask.to(self.device), float("-inf"))

        # Detect forced pass for each side
        force_pass_1 = (action_mask[:, 0].sum(dim=1) == 1) & (action_mask[:, 0, 0] == 1)
        force_pass_2 = (action_mask[:, 1].sum(dim=1) == 1) & (action_mask[:, 1, 0] == 1)

        actions = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            if force_pass_1[i] and force_pass_2[i]:
                # Both forced pass, action = (0,0)
                actions[i, 0] = 0
                actions[i, 1] = 0
            elif force_pass_1[i] and not force_pass_2[i]:
                # Force pass for side 1, mask out 0 in logits2, sample for side 2
                actions[i, 0] = 0
                logits2 = logits[i, 1].clone()
                logits2[0] = float("-inf")  # mask out action 0
                dist2 = Categorical(logits=logits2)
                actions[i, 1] = dist2.sample()
            elif force_pass_2[i] and not force_pass_1[i]:
                # Force pass for side 2, mask out 0 in logits1, sample for side 1
                actions[i, 1] = 0
                logits1 = logits[i, 0].clone()
                logits1[0] = float("-inf")  # mask out action 0
                dist1 = Categorical(logits=logits1)
                actions[i, 0] = dist1.sample()
            else:
                # Neither forced pass
                logits_side1 = logits[i, 0]
                dist1 = Categorical(logits=logits_side1)
                action1 = dist1.sample()

                # Mask out switches and tera in logits2, then sample
                logits2 = logits[i, 1].clone()
                if 1 <= action1 <= 6:
                    logits2[action1] = float("-inf")
                elif 27 <= action1 <= 46:
                    logits2[27:46] = float("-inf")

                dist2 = Categorical(logits=logits2)
                action2 = dist2.sample()

                actions[i, 0] = action1
                actions[i, 1] = action2

        return actions


if __name__ == "__main__":
    net = PseudoPolicy(observation_dim=300, action_dim=ACT_SIZE)
    action_mask = torch.ones((1, 2, ACT_SIZE))
    res = net.forward(torch.rand((1, 2, 5, 30)), action_mask)
    print(res)
