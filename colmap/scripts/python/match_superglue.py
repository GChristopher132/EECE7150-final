import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

def setup_path(superglue_path):
    sys.path.append(str(Path(superglue_path).resolve()))

def main():
    parser = argparse.ArgumentParser(
        description='SuperGlue feature matching for COLMAP',
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
        '--superglue_path', type=str, required=True,
        help='Path to the SuperGluePretrainedNetwork repository')
    
    parser.add_argument(
        '--match_threshold', type=float, default=0.15,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--weights', type=str, default='indoor', choices=['indoor', 'outdoor'],
        help='SuperGlue weights (indoor or outdoor)')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints to use for matching (-1 for all)')
    
    args = parser.parse_args()
    
    # setup path
    setup_path(args.superglue_path)
    
    try:
        from models.superglue import SuperGlue
    except ImportError:
        print("Could not import SuperGlue models. Check --superglue_path.")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    
    config = {
        'superglue': {
            'weights': args.weights,
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_threshold,
        }
    }
    
    model = SuperGlue(config.get('superglue', {})).eval().to(device)
    
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
            for i, d in enumerate([d0, d1]):
                kpts = d['keypoints']
                desc = d['descriptors']
                scores = d['scores']
                shape = d['shape']
                
                # Limit keypoints to avoid OOM
                if args.max_keypoints > 0 and len(kpts) > args.max_keypoints:
                    # Sort by score and take top k
                    # scores is (N,)
                    indices = np.argsort(scores)[::-1][:args.max_keypoints]
                    kpts = kpts[indices]
                    desc = desc[indices, :] # (N, D)
                    scores = scores[indices]
                
                data[f'keypoints{i}'] = torch.from_numpy(kpts).float().unsqueeze(0).to(device)
                data[f'descriptors{i}'] = torch.from_numpy(desc.T).float().unsqueeze(0).to(device)
                data[f'scores{i}'] = torch.from_numpy(scores).float().unsqueeze(0).to(device)
                # dummy image for shape
                data[f'image{i}'] = torch.empty((1, 1, shape[0], shape[1])).to(device)

            # inference
            try:
                with torch.no_grad():
                    pred = model(data)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: OOM for pair {name0}-{name1}, skipping. Try lowering --max_keypoints.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
            matches0 = pred['matches0'][0].cpu().numpy()
            
            # filter valid matches
            valid = matches0 > -1
            matches0_idx = np.where(valid)[0]
            matches1_idx = matches0[valid]
            
            # write to file
            f_out.write(f"{name0} {name1}\n")
            for idx0, idx1 in zip(matches0_idx, matches1_idx):
                f_out.write(f"{idx0} {idx1}\n")
            f_out.write("\n")
            
            # Clear cache to prevent OOM
            if device == 'cuda':
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
