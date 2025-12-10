import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import kornia

def main():
    parser = argparse.ArgumentParser(
        description='LightGlue feature matching for COLMAP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--pairs_file', type=str, required=True,
        help='Path to the file containing image pairs')
    parser.add_argument(
        '--features_dir', type=str, required=True,
        help='Path to the directory containing .npz feature files')
    parser.add_argument(
        '--output_file', type=str, required=True,
        help='Path to the output matches file')
    
    parser.add_argument(
        '--match_threshold', type=float, default=0.0,
        help='LightGlue match threshold')
    parser.add_argument(
        '--filter_threshold', type=float, default=0.1,
        help='LightGlue filter threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--features', type=str, default='superpoint', 
        choices=['superpoint', 'disk', 'aliked', 'sift'],
        help='Feature type used (superpoint, disk, aliked, sift)')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints to use for matching (-1 for all)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    
    # Initialize LightGlue
    matcher = kornia.feature.LightGlue(features=args.features, filter_threshold=args.filter_threshold).to(device)
    matcher.eval()
    
    pairs_path = Path(args.pairs_file)
    features_dir = Path(args.features_dir)
    output_path = Path(args.output_file)
    
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        sys.exit(1)
        
    with open(pairs_path, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines()]
        
    print(f"Processing {len(pairs)} pairs...")
    
    with open(output_path, 'w') as f_out:
        for name0, name1 in tqdm(pairs):
            npz0 = features_dir / (name0 + ".npz")
            npz1 = features_dir / (name1 + ".npz")
            
            if not npz0.exists() or not npz1.exists():
                print(f"Features not found for {name0} or {name1}")
                continue
            
            d0 = np.load(npz0)
            d1 = np.load(npz1)
            
            # prepare input
            data = {}
            
            # Helper to process each image's features
            def process_features(d, i):
                kpts = d['keypoints']
                desc = d['descriptors']
                scores = d['scores']
                shape = d['shape'] # (H, W)
                
                # Limit keypoints if requested
                if args.max_keypoints > 0 and len(kpts) > args.max_keypoints:
                    indices = np.argsort(scores)[::-1][:args.max_keypoints]
                    kpts = kpts[indices]
                    desc = desc[indices, :]
                    scores = scores[indices]
                
                # Convert to tensor
                kpts_t = torch.from_numpy(kpts).float().unsqueeze(0).to(device)
                desc_t = torch.from_numpy(desc.T).float().unsqueeze(0).to(device) # (1, D, N) or (1, N, D)?
                # Kornia LightGlue expects descriptors as (B, N, D)
                desc_t = torch.from_numpy(desc).float().unsqueeze(0).to(device) # (1, N, D)
                
                return kpts_t, desc_t, shape

            kpts0, desc0, shape0 = process_features(d0, 0)
            kpts1, desc1, shape1 = process_features(d1, 1)
            
            input_dict = {
                "image0": {
                    "keypoints": kpts0,
                    "descriptors": desc0,
                    "image_size": torch.tensor([shape0[1], shape0[0]], device=device).unsqueeze(0),
                },
                "image1": {
                    "keypoints": kpts1,
                    "descriptors": desc1,
                    "image_size": torch.tensor([shape1[1], shape1[0]], device=device).unsqueeze(0),
                }
            }
            
            # inference
            try:
                with torch.no_grad():
                    result = matcher(input_dict)
                    
                    if isinstance(result, dict):
                        matches0 = result['matches0'] # (B, N0) with indices of keypoints1, -1 for no match
                        scores = result['scores'] # (B, N0)
                    else:
                        print("Unknown output format from LightGlue")
                        continue

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: OOM for pair {name0}-{name1}, skipping. Try lowering --max_keypoints.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
            matches0 = matches0[0].cpu().numpy()
            scores = scores[0].cpu().numpy()
            
            # filter valid matches
            valid = matches0 > -1
            if args.match_threshold > 0:
                valid = valid & (scores > args.match_threshold)
                
            matches0_idx = np.where(valid)[0]
            matches1_idx = matches0[valid]
            
            # write to file
            f_out.write(f"{name0} {name1}\n")
            for idx0, idx1 in zip(matches0_idx, matches1_idx):
                f_out.write(f"{idx0} {int(idx1)}\n")
            f_out.write("\n")
            
            # Clear cache to prevent OOM
            if device == 'cuda':
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
