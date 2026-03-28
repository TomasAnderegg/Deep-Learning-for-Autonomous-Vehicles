import os
import gdown
import zipfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

COMMAND_MAP = {'forward': 0, 'left': 1, 'right': 2}

class NuplanDataLoader:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.file_urls = {
            'train': "https://drive.google.com/uc?id=1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr",
            'val': "https://drive.google.com/uc?id=1wtmT_vH9mMUNOwrNOMFP6WFw6e8rbOdu",
            'test': "https://drive.google.com/uc?id=1G9xGE7s-Ikvvc2-LZTUyuzhWAlNdLTLV"
        }
        
        # Download data if requested and not already present
        self.download_dataset()
    
    def download_dataset(self):
        print("Downloading and extracting nuPlan dataset...")
        
        for split, url in self.file_urls.items():
            output_zip = os.path.join(self.data_dir, f"dlav_{split}.zip")
            extracted_dir = os.path.join(self.data_dir, f"dlav_{split}")
            
            # Check if already extracted
            if os.path.exists(extracted_dir):
                print(f"{split} data already exists at {extracted_dir}")
                continue
            
            # Download if zip doesn't exist
            if not os.path.exists(output_zip):
                print(f"Downloading {split} data...")
                gdown.download(url, output_zip, quiet=False)

            # Extract
            print(f"Extracting {split} data...")
            with zipfile.ZipFile(output_zip , 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            print(f"{split} data downloaded and extracted successfully")

class AugmentedNuPlanDataset(Dataset):
    def __init__(self, data_path, test=False, include_dynamics=False, augment_prob=0.5):
        self.data_path = data_path
        self.test = test
        self.include_dynamics = include_dynamics
        self.augment_prob = augment_prob if not test else 0.0
        
        if test:
            self.samples = [os.path.join(data_path, fn) for fn in sorted([f for f in os.listdir(data_path) if f.endswith(".pkl")], 
                                                                          key=lambda fn: int(os.path.splitext(fn)[0]))]
        else:
            self.samples = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        
        # Define normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
        
        # Define transform
        self.augmentations = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),   
        ])
 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load pickle file
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        cmd = COMMAND_MAP[data['driving_command']]
        driving_command = torch.tensor(cmd, dtype=torch.long)
        
        # Extract camera, history and future features
        camera = torch.FloatTensor(data['camera']).permute(2, 0, 1) / 255.0
        history = torch.FloatTensor(data['sdc_history_feature'])
        # depth = torch.FloatTensor(data['depth']).permute(2, 0, 1) / 255.0
        # semantic_label = torch.LongTensor(data['semantic_label'])
        if not self.test:
            future = torch.FloatTensor(data['sdc_future_feature'])

        # Normalize camera
        camera = self.normalize(camera)  
        
        # Apply random flip to camera and history features
        if random.random() < self.augment_prob:
            camera = self.augmentations(camera)
            camera = torch.flip(camera, dims=[2]) # Flip horizontally
            # depth = torch.flip(depth, dims=[2])
            # semantic_label = torch.flip(semantic_label, dims=[1])
            history[:, 1] = -history[:, 1] # Flip y-coordinates
            history[:, 2] = -history[:, 2] # Flip heading-coordinates
            if not self.test:
                future[:, 1] = -future[:, 1]
                future[:, 2] = -future[:, 2]
            if cmd == 1: cmd = 2        # Quand on flippe l'image, tout ce qui est a gauche/droite doit etre inverse
            elif cmd == 2: cmd = 1 
            driving_command = torch.tensor(cmd, dtype=torch.long)   
                
        if self.include_dynamics:
            history = self.compute_dynamics_features(history)
                
        # Create sample dictionary
        if self.test:
            sample = {
                'camera': camera,
                'history': history,
                'driving_command': driving_command,
                # 'depth': depth,
                # 'semantic_label': semantic_label,
            }
        else:
            sample = {
                'camera': camera,
                'history': history,
                'driving_command': driving_command,
                # 'depth': depth,
                # 'semantic_label': semantic_label,
                'future': future,
            }
        return sample
    
    def compute_dynamics_features(self, trajectory):
        positions = trajectory[:, :2]
        
        # Initialize new features
        timesteps = trajectory.shape[0]
        dynamics_features = torch.zeros((timesteps, 5), dtype=trajectory.dtype)
        
        if timesteps >= 2: # Need at least 2 points for velocity
            # Compute velocity (v_x, v_y)
            velocities = (positions[1:] - positions[:-1])
            velocities = torch.cat([velocities[:1], velocities], dim=0)
            
            # Velocity magnitude
            vel_magnitude = torch.norm(velocities, dim=1, keepdim=True)
            
            # Add velocities to features
            dynamics_features[:, 0:2] = velocities
            dynamics_features[:, 2:3] = vel_magnitude
        
        if timesteps >= 3: # Need at least 3 points for acceleration
            # Compute acceleration (a_x, a_y)
            accelerations = (velocities[1:] - velocities[:-1])
            accelerations = torch.cat([accelerations[:1], accelerations], dim=0)
            
            # Add accelerations to features
            dynamics_features[:, 3:5] = accelerations
            
        augmented_trajectory = torch.cat([trajectory, dynamics_features], dim=1)
        
        return augmented_trajectory
            
def visualize_samples(dataset, num_samples=4):
    
    # Define the inverse normalization transform
    inverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    # Process each sample
    for i, idx in enumerate(sample_indices):
        data = dataset[idx]

        # Plot camera view (top row)
        camera_img = inverse_normalize(data["camera"]).clamp(0, 1).permute(1, 2, 0)
        axes[0, i].imshow(camera_img)
        axes[0, i].set_title(f"Camera View {i+1}")
        axes[0, i].axis("off")

        # Plot trajectory (bottom row)
        history = data["history"]
        future = data["future"]

        axes[1, i].plot(history[:, 0], history[:, 1], "o-", color="gold", linewidth=2, label="Past")
        axes[1, i].plot(future[:, 0], future[:, 1], "o-", color="green", linewidth=2, label="Future")
        axes[1, i].set_title(f"Trajectory {i+1} — {['forward','left','right'][data['driving_command'].item()]}")
        axes[1, i].legend(loc='upper right')
        axes[1, i].axis("equal")
        axes[1, i].grid(True, linestyle='--', alpha=0.7)
        
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

def get_data_paths(data_dir):
    return {
        'train': os.path.join(data_dir, "train"),
        'val': os.path.join(data_dir, "val"),
        'test': os.path.join(data_dir, "test_public")
    }
    
def main():

    # Initialize data loader
    data_dir = "."
    NuplanDataLoader(data_dir=data_dir)
    data_paths = get_data_paths(data_dir=data_dir)

    # Visualize samples from the training set
    dataset = AugmentedNuPlanDataset(data_paths['train'], test=False, include_dynamics=True, augment_prob=0.5)
    visualize_samples(dataset, num_samples=4)
    
if __name__ == "__main__":
    main()
    