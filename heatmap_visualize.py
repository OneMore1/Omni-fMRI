import os
import argparse
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from finetune import create_model, load_config, ADNIDataset, HCPtaskDataset, fMRITaskDataset1


AFFINE_TARGET = np.array([
    [-2,  0,  0,   96],
    [ 0,  2,  0, -112],
    [ 0,  0,  2,  -90],
    [ 0,  0,  0,    1]
])


class IntegratedGradients:
    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor, target_class=None, steps=20, task_type='classification'):

        self.model.eval()
        input_tensor.requires_grad = True
        
        baseline = torch.zeros_like(input_tensor)
        
        alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
        
        total_gradients = torch.zeros_like(input_tensor)
        
        print(f"Running Integrated Gradients with {steps} steps...")
        
        for alpha in alphas:
            alpha_val = alpha.view(1, 1, 1, 1, 1) 
            
            current_input = baseline + alpha_val * (input_tensor - baseline)
            current_input = current_input.detach().requires_grad_(True)
            
            output = self.model(current_input)
            self.model.zero_grad()
            
            if task_type == 'classification':
                if target_class is None:
                    target_class = output.argmax(dim=1)
                
                if output.shape[0] > 1:
                    score = torch.gather(output, 1, target_class.unsqueeze(1)).sum()
                else:
                    score = output[:, target_class].sum()
            else: # Regression
                score = output.sum()

            score.backward()

            if current_input.grad is not None:
                total_gradients += current_input.grad
            
        avg_gradients = total_gradients / steps
        
        #  = (Input - Baseline) * Avg_Gradients
        ig = (input_tensor - baseline) * avg_gradients
        
        return ig


def save_nifti_zscore(data_numpy, output_path, affine=None, do_zscore=True):

    if affine is None:
        affine = AFFINE_TARGET
        
    data = np.nan_to_num(data_numpy)

    if do_zscore:
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 1e-8:
            print(f"  [Warning] Data std is close to 0 at {os.path.basename(output_path)}. Saving as zeros.")
            final_data = np.zeros_like(data)
        else:
            final_data = (data - mean_val) / std_val
            
        # print(f"  [Info] Z-Score applied. Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    else:
        final_data = data

    img = nib.Nifti1Image(final_data.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Saved: {output_path}")

def process_and_save(attributions, original_input, save_dir, prefix, sample_idx):

    attr_data = attributions[0].detach().cpu().numpy()
    input_data = original_input[0].detach().cpu().numpy()
    

    heatmap_no_avg = attr_data.transpose(1, 2, 3, 0)
    input_nii = input_data.transpose(1, 2, 3, 0)
    
    heatmap_time_raw = np.mean(np.abs(attr_data), axis=0) # (H, W, D)

    heatmap_time_avg = gaussian_filter(heatmap_time_raw, sigma=1.0)
    heatmap_no_avg_smooth = gaussian_filter(heatmap_no_avg, sigma=1.0)

    save_nifti_zscore(
        input_nii, 
        os.path.join(save_dir, f"{prefix}_idx{sample_idx}_input.nii.gz"),
        do_zscore=False 
    )
    
    save_nifti_zscore(
        heatmap_no_avg_smooth, 
        os.path.join(save_dir, f"{prefix}_idx{sample_idx}_heatmap_no_avg.nii.gz"),
        do_zscore=True
    )
    
    save_nifti_zscore(
        heatmap_time_avg, 
        os.path.join(save_dir, f"{prefix}_idx{sample_idx}_heatmap_time_avg.nii.gz"),
        do_zscore=True
    )


def main():
    parser = argparse.ArgumentParser(description='(IG + ZScore)')
    parser.add_argument('--config', type=str, default='', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default="", help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--target_class', type=int, default=1, help='Target class index (optional)')
    parser.add_argument('--steps', type=int, default=50, help='Integration steps for IG')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...")
    model = create_model(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print("Loading Dataset...")


    test_dataset = HCPtaskDataset(txt_path="")
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    task_type = config['task']['task_type']
    print(f"Task Type: {task_type}")

    print(f"Starting Integrated Gradients (steps={args.steps})...")
    print("Output will be Z-Scored and saved with target Affine.")
    
    ig_solver = IntegratedGradients(model)
    
    count = 0
    for samples, labels in test_loader:
        if count >= args.num_samples:
            break
        
        print(f"Processing sample {count}...")

        samples = samples.to(device) 

        attributions = ig_solver.generate(
            samples, 
            target_class=args.target_class if args.target_class is not None else (labels[0].item() if task_type=='classification' else None),
            steps=args.steps,
            task_type=task_type
        )

        process_and_save(attributions, samples, args.output_dir, "IG", count)
        
        count += 1

    print(f"\nVisualization complete. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()