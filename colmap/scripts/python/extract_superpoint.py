import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# add SuperGluePretrainedNetwork to path
def setup_path(superglue_path):
    sys.path.append(str(Path(superglue_path).resolve()))

def main():
    parser = argparse.ArgumentParser(
        description='SuperPoint feature extraction for colmap',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='path to the directory containing images')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='path to the directory to save feature files')
    parser.add_argument(
        '--superglue_path', type=str, required=True,
        help='path to the SuperGluePretrainedNetwork repo')
    
    parser.add_argument(
        '--max_keypoints', type=int, default=2048,
        help='max number of keypoints detected by Superpoint')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.0005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1600],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    
    args = parser.parse_args()
    
    # setup path
    setup_path(args.superglue_path)
    
    try:
        from models.superpoint import SuperPoint
        from models.utils import read_image
    except ImportError as e:
        print(f"Could not import SuperPoint models. Check --superglue_path. Error: {e}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    
    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        }
    }
    
    model = SuperPoint(config.get('superpoint', {})).eval().to(device)
    
    input_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(list(input_dir.glob(ext)))
    images.sort()
    
    if len(images) == 0:
        print(f"No images found in {input_dir}")
        sys.exit(1)
        
    print(f"Processing {len(images)} images...")
    
    for img_path in tqdm(images):
        # read image
        image, inp, scales = read_image(
            img_path, device, args.resize, 0, False)
        
        if image is None:
            print(f"Failed to read image {img_path}")
            continue
            
        # inference
        with torch.no_grad():
            pred = model({'image': inp})
            
        # unpack
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts = pred['keypoints']
        desc = pred['descriptors'].T # (N, D)
        scores = pred['scores']
        
        # rescale keypoints
        kpts[:, 0] *= scales[0]
        kpts[:, 1] *= scales[1]
        
        # save as npz
        npz_path = output_dir / (img_path.name + ".npz")
        np.savez(npz_path, keypoints=kpts, descriptors=desc, scores=scores, shape=image.shape[:2])
        
        # write to txt
        txt_path = output_dir / (img_path.name + ".txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"{len(kpts)} 128\n")
            for i in range(len(kpts)):
                x, y = kpts[i]
                # colmap expects 0-255 descriptors
                d = np.clip(desc[i] * 512.0, 0, 255)
                
                f.write(f"{x:.6f} {y:.6f} 1.0 0.0")
                for val in d:
                    f.write(f" {int(val)}")
                f.write("\n")
        
        # Clear cache to prevent OOM
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
