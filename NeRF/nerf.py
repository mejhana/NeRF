import torch 
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, emb_dim_pos=10, emb_dim_dir=4, hidden_dim=256):
        super(NeRF, self).__init__()
        self.hid1 = nn.Sequential(nn.Linear(emb_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density: 
            # inputs: position (x,y,z) after positional encoding + hidden representation (hidden_dim), 
            # outputs: hidden representation (hidden_dim) + density (σ)
        self.fc_den = nn.Sequential(nn.Linear(emb_dim_pos * 6 + 3 + hidden_dim , hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), ) 
        # color: 
            # inputs: hidden representation with density (σ), direction (Φ,ψ) after positional encoding, 
            # outputs: color (RGB)

        self.hid2 = nn.Sequential(nn.Linear(emb_dim_dir * 6 + 3 + hidden_dim , hidden_dim // 2), nn.ReLU(), )
        self.fc_rgb = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        # for later! 
        self.emb_dim_pos = emb_dim_pos
        self.emb_dim_dir = emb_dim_dir
        self.relu = nn.ReLU()
        
    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
    
    def forward(self, pos, dir):
        emb_pos = self.positional_encoding(pos, self.emb_dim_pos)
        emb_dir = self.positional_encoding(dir, self.emb_dim_dir)

        h = self.hid1(emb_pos)
        den = self.fc_den(torch.cat((h, emb_pos), dim=1)) # den: [batch_size, hidden_dim + 1]
        h, sigma = den[:, :-1], self.relu(den[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.hid2(torch.cat((h, emb_dir), dim=1)) # h: [batch_size, hidden_dim // 2]
        col = self.fc_rgb(h) # c: [batch_size, 3]
        return col, sigma



    

    



        
        
            
        