import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.PPMIDataset import PPMIDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Set training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4

    # Initialize dataset
    dataset = PPMIDataset(
        processed_data_dir="processed-data",  # Path to your processed data directory
        img_size=224,
        live_loading=True,
        train=True,
        one_hot_tabular=False,
        corruption_rate=0.2
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type != 'mps' else 0,  # MPS doesn't work well with multiprocessing
        pin_memory=device.type == 'cuda'  # Only pin memory for CUDA
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Tabular input size: {dataset.get_input_size()}")

    # TODO: Initialize your model here
    # model = YourModel()
    # model.to(device)

    # TODO: Set up optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = YourLossFunction()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                imaging_views = [view.to(device) for view in batch['imaging_views']]
                tabular_views = [view.to(device) for view in batch['tabular_views']]

                # TODO: Your training step here
                # optimizer.zero_grad()
                # output = model(imaging_views, tabular_views)
                # loss = criterion(output)
                # loss.backward()
                # optimizer.step()
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}/{NUM_EPOCHS}, Batch {batch_idx}/{len(dataloader)}")

            except RuntimeError as e:
                if "MPS" in str(e):
                    print(f"MPS error encountered: {e}")
                    print("Falling back to CPU for this batch...")
                    # Fallback to CPU for this batch
                    imaging_views = [view.to('cpu') for view in batch['imaging_views']]
                    tabular_views = [view.to('cpu') for view in batch['tabular_views']]
                    # TODO: Move model to CPU, process batch, then move back to MPS
                else:
                    raise e

        # Print epoch summary
        print(f"Epoch {epoch}/{NUM_EPOCHS} complete")

        # TODO: Add validation step here
        # TODO: Add model checkpointing here

if __name__ == "__main__":
    main() 