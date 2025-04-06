import satlaspretrain_models
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

# model
def get_model(device):
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2, device=device).to(device)
    return model

# training loop for forest datasets
def forest_finetune(model, loader, epochs, model_name='jimmy', lr=1e-4, device='cuda'):

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # training loop
    print(f"Initiating {model_name}'s training session ...")
    for epoch in range(epochs):

        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        epoch_loss = 0
        lossi = []

        for inputs, masks in progress_bar:
            inputs, masks = inputs.float().to(device), masks.squeeze(1).long().to(device)
            optimizer.zero_grad()
            outs, loss = model(inputs, masks)
            loss.backward()
            optimizer.step()
            
            loss_value = loss.detach().cpu().item()
            lossi.append(loss_value)
            epoch_loss += loss_value
            del inputs, masks, outs, loss_value
   
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/{model_name}_epoch{epoch+1}.pth')

        print(f'Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.4f}')
        plt.plot(lossi)
        plt.savefig('losses.png')
    print(f"{model_name}'s training session has concluded.")
