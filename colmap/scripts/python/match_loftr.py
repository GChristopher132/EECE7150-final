import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import kornia
from kornia.feature import LoFTR
from tqdm import tqdm
import cv2

def main():
    parser = argparse.ArgumentParser(
        description='LoftR feature matching for COLMAP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--pairs_file', type=str, required=True,
        help='Path to the file containing image pairs')
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Path to the directory containing images')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path to the directory to save feature and match files')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='LoftR match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--weights', type=str, default='outdoor', choices=['indoor', 'outdoor'],
        help='LoftR weights (indoor or outdoor)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    
    matcher = LoFTR(pretrained=args.weights).to(device)
    matcher.eval()
    
    pairs_path = Path(args.pairs_file)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        sys.exit(1)
        
    with open(pairs_path, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines()]
        
    print(f"Processing {len(pairs)} pairs...")
    
    # Store all keypoints for each image
    # image_name -> list of (x, y) tuples
    image_keypoints = {}
    
    # Store matches as (image1_name, image2_name) -> (indices1, indices2)
    matches_data = []
    
    for name0, name1 in tqdm(pairs):
        path0 = image_dir / name0
        path1 = image_dir / name1
        
        if not path0.exists() or not path1.exists():
            print(f"Image not found: {name0} or {name1}")
            continue
            
        # Load and resize images
        # LoftR works best with resized images (e.g. 640 or 840) to avoid OOM and for speed
        # We'll use kornia's resize or just load at a smaller scale if possible, 
        # but kornia.io.load_image loads full res.
        
        img0 = kornia.io.load_image(str(path0), kornia.io.ImageLoadType.GRAY8).float() / 255.0
        img1 = kornia.io.load_image(str(path1), kornia.io.ImageLoadType.GRAY8).float() / 255.0
        
        # Resize if too large (simple max dimension check)
        max_dim = 840 # Standard for LoftR indoor/outdoor
        scale0 = 1.0
        if max(img0.shape[-2:]) > max_dim:
            scale0 = max_dim / max(img0.shape[-2:])
            img0 = kornia.geometry.transform.resize(img0, (int(img0.shape[-2]*scale0), int(img0.shape[-1]*scale0)))
            
        scale1 = 1.0
        if max(img1.shape[-2:]) > max_dim:
            scale1 = max_dim / max(img1.shape[-2:])
            img1 = kornia.geometry.transform.resize(img1, (int(img1.shape[-2]*scale1), int(img1.shape[-1]*scale1)))

        
        img0 = img0.to(device)
        img1 = img1.to(device)
        
        input_dict = {"image0": img0[None], "image1": img1[None]}
        
        with torch.no_grad():
            correspondences = matcher(input_dict)
            
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()
        
        # Filter by threshold
        mask = confidence >= args.match_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        
        if len(mkpts0) == 0:
            continue
            
        # Scale keypoints back to original resolution
        if 'scale0' in locals():
            mkpts0 = mkpts0 / scale0
        if 'scale1' in locals():
            mkpts1 = mkpts1 / scale1

        # Convert to tensors for distance calculation if not already
        # mkpts are numpy arrays currently
        t_mkpts0 = torch.from_numpy(mkpts0).float().to(device)
        t_mkpts1 = torch.from_numpy(mkpts1).float().to(device)

        def get_indices(name, kpts, threshold=2.0):
            if name not in image_keypoints:
                image_keypoints[name] = kpts
                return np.arange(len(kpts))
            
            existing = image_keypoints[name]
            # Calculate distances
            # existing: (N, 2), kpts: (M, 2)
            # We want to find for each kpt, if it exists in existing
            
            # Process in chunks to avoid huge memory usage if N*M is large
            # But here M is small (~2k), N grows.
            
            dists = torch.cdist(kpts, existing) # (M, N)
            min_dists, min_indices = torch.min(dists, dim=1)
            
            mask_exists = min_dists < threshold
            
            indices = torch.zeros(len(kpts), dtype=torch.long, device=device)
            indices[mask_exists] = min_indices[mask_exists]
            
            # Handle new points
            mask_new = ~mask_exists
            if mask_new.any():
                new_points = kpts[mask_new]
                start_idx = len(existing)
                num_new = len(new_points)
                
                # Append new points
                image_keypoints[name] = torch.cat([existing, new_points], dim=0)
                
                # Assign new indices
                indices[mask_new] = torch.arange(start_idx, start_idx + num_new, device=device)
                
            return indices.cpu().numpy()

        indices0 = get_indices(name0, t_mkpts0)
        indices1 = get_indices(name1, t_mkpts1)
        
        matches_data.append((name0, name1, indices0, indices1))

    print("Consolidating keypoints and writing output...")
    
    # Write feature files
    for name, kpts_tensor in image_keypoints.items():
        # Convert to numpy for easier handling
        kpts = kpts_tensor.cpu().numpy()
        
        # Create dummy descriptors (COLMAP needs them)
        # We use 128-dim zero vectors or random, doesn't matter as we provide matches
        num_kpts = len(kpts)
        desc = np.zeros((num_kpts, 128), dtype=np.uint8)
        
        # Write to .txt format for import
        txt_path = output_dir / (name + ".txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"{num_kpts} 128\n")
            for i in range(num_kpts):
                x, y = kpts[i]
                # x, y, scale, orientation
                f.write(f"{x:.6f} {y:.6f} 1.0 0.0")
                for val in desc[i]:
                    f.write(f" {val}")
                f.write("\n")
                
    # Write matches file
    matches_path = output_dir / "matches.txt"
    with open(matches_path, 'w') as f:
        for name0, name1, idx0, idx1 in matches_data:
            f.write(f"{name0} {name1}\n")
            for i0, i1 in zip(idx0, idx1):
                f.write(f"{i0} {i1}\n")
            f.write("\n")

if __name__ == '__main__':
    main()
