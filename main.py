import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from NeRF.render import render_rays
from NeRF.nerf import NeRF
# from NeRF.data_processing import viz_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def test(model, device, hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.axis('off')
    plt.gca().set_facecolor('black')
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

def train(nerf_model, optimizer, scheduler, data_loader, device, hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    
    save_every=1

    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_loss = 0
        
        # check if the model for the current epoch exists
        if os.path.exists(f'nerf_model_epoch_{epoch + 1}.pth'):
            nerf_model.load_state_dict(torch.load(f'nerf_model_epoch_{epoch + 1}.pth'))
            continue

        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            training_loss.append(loss.item())

        print(f'Loss for {epoch}: {epoch_loss}')
        scheduler.step()

        if (epoch + 1) % save_every == 0:
            torch.save(nerf_model.state_dict(), f'nerf_model_epoch_{epoch + 1}.pth')
            
        # vizualize 200//epochs images for every epoch; 
        n_images = len(testing_dataset)//H//W
        images_to_viz = n_images // nb_epochs
        start_index = epoch * images_to_viz
        end_index = (epoch + 1) * images_to_viz
        for img_index in range(start_index, end_index):  # Visualize a few images
            test(nerf_model, device, hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)

    return training_loss

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    print('Loading data...')
    training_dataset = torch.from_numpy(np.load('data/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('data/testing_data.pkl', allow_pickle=True))
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    
    # # viz 
    # viz_dataset(training_dataset)
    # viz_synth()

    # model
    emb_dim_pos = 10
    emb_dim_dir = 4
    hidden_dim = 256
    model = NeRF(emb_dim_pos=emb_dim_pos, emb_dim_dir=emb_dim_dir, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    
    # training
    print(f'Training model on {device}...')
    train(model, optimizer, scheduler, data_loader, device, hn=2, hf=6, nb_epochs=16, nb_bins=192, H=400,
          W=400)