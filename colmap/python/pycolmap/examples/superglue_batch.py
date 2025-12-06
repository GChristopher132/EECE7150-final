"""
Batch SuperPoint + SuperGlue matcher for COLMAP

This script extracts keypoints (SuperPoint when available, otherwise OpenCV SIFT),
runs SuperGlue matching on pairs of images and writes a COLMAP-compatible "pairs"
text file with matches.

Notes:
 - This script tries to import common SuperGlue/SuperPoint repo layouts. You may
   need to adapt `load_superglue_module` or provide a thin wrapper that exposes
   a consistent callable API.
 - PyTorch + GPU recommended for speed. If not available, CPU fallback will be used.
 - You can provide an explicit `--pair_list` (one pair per line: imgA imgB), or
   use `--exhaustive` to match all pairs (costly), or pass `--pairs_from_filelist`
   to match consecutive pairs from an image list.

Output format (pairs file):
  Each line: <basename(imgA)> <basename(imgB)> <num_matches> i0 j0 i1 j1 ...
  where iN/jN are 0-based keypoint indices into the detected keypoints arrays.

Example:
  python -m pycolmap.examples.superglue_batch \
    --image_dir /path/to/images \
    --superglue_dir /path/to/SuperGluePretrainedNetwork \
    --output_pairs /tmp/pairs.txt --device cuda --exhaustive
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict

# Ensure the pycolmap package is importable when running the script directly
# Directory layout: colmap/python/pycolmap/examples/this_file -> go up 3 levels to colmap/python
_SCRIPT_DIR = os.path.dirname(__file__)
_PYTHON_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pycolmap
except Exception:
    pycolmap = None


def load_superglue_module(superglue_dir: str):
    """Try to import SuperGlue modules from a directory.

    Returns either a module object with `SuperGlue` class or a tuple
    (superpoint_mod, superglue_mod) depending on available layout.
    """
    if superglue_dir is None or superglue_dir == "":
        raise RuntimeError("`--superglue_dir` must point to your SuperGlue repo")
    sys.path.insert(0, superglue_dir)
    # Try several common patterns
    try:
        import importlib
        import models
        # Ensure submodules are attached as attributes for downstream logic.
        if not hasattr(models, 'superpoint'):
            try:
                models.superpoint = importlib.import_module('models.superpoint')
            except Exception:
                pass
        if not hasattr(models, 'superglue'):
            try:
                models.superglue = importlib.import_module('models.superglue')
            except Exception:
                pass
        # Return the top-level models module. Caller will access models.superpoint/models.superglue
        if hasattr(models, 'superpoint') and hasattr(models, 'superglue'):
            return models
    except Exception:
        pass

    try:
        import superpoint
        import superglue
        return (superpoint, superglue)
    except Exception:
        pass

    raise RuntimeError("Could not import SuperGlue modules from: %s" % superglue_dir)


def extract_with_pycolmap_or_sift(image_path: str, max_kp: int = 0):
    """Fallback extractor: try pycolmap, else OpenCV SIFT (grayscale).

    Returns keypoints (N,2 float) and descriptors (N,D float)
    """
    if pycolmap is not None:
        try:
            feats = pycolmap.extract_features(image_path)
            kps = np.array([[k.x, k.y] for k in feats.keypoints], dtype=np.float32)
            desc = np.array(feats.descriptors, dtype=np.float32)
            return kps, desc
        except Exception:
            # Fall through to OpenCV
            pass

    if cv2 is None:
        raise RuntimeError("No feature extractor available: install pycolmap or OpenCV")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    sift = cv2.SIFT_create(nfeatures=max_kp if max_kp > 0 else 0)
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,128), dtype=np.float32)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    return pts, desc.astype(np.float32)


def extract_with_repo_superpoint(sp_net, image_path: str, device: Optional[str] = "cpu"):
    """Extract keypoints/descriptors using the SuperPoint model from the repo.

    Returns (kps Nx2 float32, desc NxD float32, scores N)
    """
    import torch
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    img_t = torch.from_numpy(img.astype('float32') / 255.0)[None, None]
    if device is not None and torch.cuda.is_available() and device.startswith('cuda'):
        sp_net = sp_net.cuda()
        img_t = img_t.cuda()

    with torch.no_grad():
        out = sp_net({'image': img_t})

    # repo outputs sometimes wrap tensors in lists/tuples
    def unwrap(x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    kps = unwrap(out.get('keypoints'))
    desc = unwrap(out.get('descriptors'))
    scores = unwrap(out.get('scores')) if 'scores' in out else None

    # Convert to numpy safely
    kps_np = kps.detach().cpu().numpy()
    # descriptors in repo are (D, N) -> convert to (N, D)
    desc_np = desc.detach().cpu().numpy()
    if desc_np.ndim == 2 and desc_np.shape[0] < desc_np.shape[1]:
        # probably (D, N) -> transpose to (N, D)
        desc_np = desc_np.T
    elif desc_np.ndim == 3:
        # sometimes wrapped in batch dim (1, D, N)
        desc_np = desc_np[0].T

    if scores is None:
        scores_np = np.ones((kps_np.shape[0],), dtype=np.float32)
    else:
        scores_np = scores.detach().cpu().numpy()

    return kps_np.astype(np.float32), desc_np.astype(np.float32), scores_np.astype(np.float32)


def instantiate_superglue(module, device: str, descriptor_dim: int):
    if torch is None:
        raise RuntimeError("PyTorch required to run SuperGlue")
    if hasattr(module, 'SuperGlue'):
        cfg = {}
        # Prefer explicit default_config, else uppercase attributes, else fallback minimal config.
        if hasattr(module, 'default_config') and isinstance(module.default_config, dict):
            cfg.update(module.default_config)
        else:
            for k in dir(module):
                if k.isupper():
                    cfg[k.lower()] = getattr(module, k)
        # Ensure descriptor_dim aligns with extracted descriptors (SuperPoint usually 256).
        if 'descriptor_dim' not in cfg or cfg['descriptor_dim'] != descriptor_dim:
            cfg['descriptor_dim'] = descriptor_dim
        # Provide safe defaults if missing.
        cfg.setdefault('match_threshold', 0.2)
        cfg.setdefault('sinkhorn_iterations', 50)
        model = module.SuperGlue(cfg)
        device_obj = torch.device('cuda' if (device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
        model.to(device_obj)
        model.eval()
        return model
    raise RuntimeError("Provided module does not expose SuperGlue class")


def load_model_weights(model, weight_candidates: List[str]):
    if torch is None:
        return
    for p in weight_candidates:
        if p and os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location='cpu')
                state = ckpt.get('state_dict', ckpt)
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"Loaded weights from {p} (missing={len(missing)}, unexpected={len(unexpected)})")
                return
            except Exception as e:
                print(f"Failed loading weights {p}: {e}")
    print("Warning: SuperGlue weights not found; matches may be poor or empty.")


def run_superglue_on_pair(
    kps0: np.ndarray,
    desc0: np.ndarray,
    scores0: np.ndarray,
    kps1: np.ndarray,
    desc1: np.ndarray,
    scores1: np.ndarray,
    model,
    device: str = 'cuda'
):
    if torch is None:
        raise RuntimeError("PyTorch required to run SuperGlue")
    if kps0.shape[0] == 0 or kps1.shape[0] == 0:
        return []
    device_obj = torch.device('cuda' if (device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
    # Keypoints (B, N, 2)
    kps0_t = torch.from_numpy(kps0)[None].float().to(device_obj)
    kps1_t = torch.from_numpy(kps1)[None].float().to(device_obj)
    # Descriptors need (B, D, N)
    if desc0.ndim == 2:
        desc0_t = torch.from_numpy(desc0.T)[None].float().to(device_obj)
    else:
        raise RuntimeError("Unexpected descriptor shape for image0")
    if desc1.ndim == 2:
        desc1_t = torch.from_numpy(desc1.T)[None].float().to(device_obj)
    else:
        raise RuntimeError("Unexpected descriptor shape for image1")
    scores0_t = torch.from_numpy(scores0)[None].float().to(device_obj)
    scores1_t = torch.from_numpy(scores1)[None].float().to(device_obj)
    with torch.no_grad():
        out = model({
            'image0': torch.zeros((1,1,1,1), device=device_obj),  # dummy
            'image1': torch.zeros((1,1,1,1), device=device_obj),
            'keypoints0': kps0_t,
            'keypoints1': kps1_t,
            'descriptors0': desc0_t,
            'descriptors1': desc1_t,
            'scores0': scores0_t,
            'scores1': scores1_t,
        })
    matches0 = out['matches0'][0].detach().cpu().numpy()
    match_scores0 = out.get('matching_scores0')
    if match_scores0 is not None:
        match_scores0 = match_scores0[0].detach().cpu().numpy()
    matches = []
    for i, j in enumerate(matches0):
        if j >= 0:
            score = float(match_scores0[i]) if match_scores0 is not None else 1.0
            matches.append((int(i), int(j), score))
    return matches


def gather_image_list(image_dir: Optional[str], image_list_file: Optional[str]) -> List[str]:
    if image_list_file:
        with open(image_list_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines
    if image_dir is None:
        raise RuntimeError("Either --image_dir or --image_list must be provided")
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
    images = [os.path.join(image_dir, p) for p in os.listdir(image_dir)
              if os.path.splitext(p)[1].lower() in exts]
    images.sort()
    return images


def parse_pair_list(pair_list_path: str, images: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    name_to_path = {os.path.basename(p): p for p in images}
    with open(pair_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a in name_to_path and b in name_to_path:
                pairs.append((name_to_path[a], name_to_path[b]))
            else:
                if os.path.exists(a) and os.path.exists(b):
                    pairs.append((a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Directory with images')
    parser.add_argument('--image_list', help='One image path per line')
    parser.add_argument('--pair_list', help='Optional pair list (basename or full path)')
    parser.add_argument('--pairs', dest='pair_list', help='Alias for --pair_list')
    parser.add_argument('--output_pairs', default='pairs.txt')
    parser.add_argument('--out_pairs_path', dest='output_pairs', help='Alias for --output_pairs')
    parser.add_argument('--superglue_dir', required=True,
                        help='Path to SuperGlue repo checkout')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_num_keypoints', type=int, default=1024)
    parser.add_argument('--exhaustive', action='store_true',
                        help='Match all pairs (costly)')
    parser.add_argument('--match_consecutive', action='store_true',
                        help='Match consecutive images only (image[i], image[i+1])')
    parser.add_argument('--min_keypoints', type=int, default=5)
    parser.add_argument('--superpoint_weights', help='Path to SuperPoint weights (.pth)')
    parser.add_argument('--superglue_weights', help='Path to SuperGlue weights (.pth)')
    args = parser.parse_args()

    images = gather_image_list(args.image_dir, args.image_list)
    if len(images) == 0:
        print('No images found')
        return

    if args.pair_list:
        pairs = parse_pair_list(args.pair_list, images)
    elif args.exhaustive:
        from itertools import combinations
        pairs = list(combinations(images, 2))
    elif args.match_consecutive:
        pairs = [(images[i], images[i+1]) for i in range(len(images)-1)]
    else:
        # default: match consecutive
        pairs = [(images[i], images[i+1]) for i in range(len(images)-1)]

    print(f'Processing {len(images)} images, {len(pairs)} pairs')

    # Load SuperGlue module
    sg_mod = load_superglue_module(args.superglue_dir)

    # If the repo provides models.superpoint / models.superglue, instantiate SuperPoint for extraction
    use_repo_sp = False
    sp_net = None
    match_mod = None
    try:
        # models module style
        if hasattr(sg_mod, 'superpoint') and hasattr(sg_mod, 'superglue'):
            models_mod = sg_mod
            sp_mod = models_mod.superpoint
            sp_net = sp_mod.SuperPoint({'nms_radius':4,'keypoint_threshold':0.005,'max_keypoints':args.max_num_keypoints})
            use_repo_sp = True
            match_mod = models_mod.superglue
            print('Using SuperPoint from repo for extraction')
        # (superpoint, superglue) tuple style
        elif isinstance(sg_mod, tuple):
            sp_mod, sg_sub = sg_mod
            sp_net = sp_mod.SuperPoint({'nms_radius':4,'keypoint_threshold':0.005,'max_keypoints':args.max_num_keypoints})
            use_repo_sp = True
            match_mod = sg_sub
            print('Using SuperPoint (separate) from repo for extraction')
    except Exception as e:
        print(f'Failed to instantiate repo SuperPoint: {e}; falling back to pycolmap/OpenCV')

    # Cache keypoints/descriptors per image. If using repo SuperPoint, also store scores.
    feats_cache: Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}

    for img_path in set([p for pair in pairs for p in pair]):
        try:
            if use_repo_sp and sp_net is not None:
                kps, desc, scores = extract_with_repo_superpoint(sp_net, img_path, device=args.device)
            else:
                kps, desc = extract_with_pycolmap_or_sift(img_path, max_kp=args.max_num_keypoints)
                scores = np.ones((kps.shape[0],), dtype=np.float32)
        except Exception as e:
            print(f'Failed to extract for {img_path}: {e}')
            kps = np.zeros((0,2), dtype=np.float32)
            desc = np.zeros((0,128), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        feats_cache[img_path] = (kps, desc, scores)
        print(f'Extracted {kps.shape[0]} keypoints from {os.path.basename(img_path)}')

    # Instantiate SuperGlue model (only if we have repo SuperPoint or user insists)
    if not use_repo_sp or match_mod is None:
        print("SuperPoint repo model not initialized; cannot run SuperGlue. Check --superglue_dir path and dependencies.")
        return
    descriptor_dim = None
    for (k, d, s) in feats_cache.values():
        if d.shape[0] > 0:
            descriptor_dim = d.shape[1]
            break
    if descriptor_dim is None:
        print("No descriptors extracted; aborting matching.")
        return
    try:
        superglue_model = instantiate_superglue(match_mod, args.device, descriptor_dim)
        load_model_weights(superglue_model, [
            args.superglue_weights,
            os.path.join(args.superglue_dir, 'models', 'weights', 'superglue_outdoor.pth'),
            os.path.join(args.superglue_dir, 'models', 'weights', 'superglue_indoor.pth'),
        ])
    except Exception as e:
        print(f"Failed to initialize SuperGlue: {e}")
        return

    # If we used repo SuperPoint, try loading weights for it too
    if use_repo_sp and sp_net is not None and torch is not None:
        try:
            load_model_weights(sp_net, [
                args.superpoint_weights,
                os.path.join(args.superglue_dir, 'weights', 'superpoint_v1.pth'),
                os.path.join(args.superglue_dir, 'models', 'weights', 'superpoint_v1.pth'),
            ])
        except Exception as e:
            print(f"Warning: could not load SuperPoint weights: {e}")

    # Run matching for each pair
    all_lines = []
    for a,b in pairs:
        kps0, desc0, scores0 = feats_cache[a]
        kps1, desc1, scores1 = feats_cache[b]
        if kps0.shape[0] < args.min_keypoints or kps1.shape[0] < args.min_keypoints:
            print(f'Skipping pair {os.path.basename(a)} {os.path.basename(b)} (too few keypoints)')
            continue
        try:
            matches = run_superglue_on_pair(kps0, desc0, scores0, kps1, desc1, scores1, superglue_model, device=args.device)
        except Exception as e:
            print(f'Error running SuperGlue on pair {a} {b}: {e}')
            matches = []
        print(f'Found {len(matches)} matches for {os.path.basename(a)} {os.path.basename(b)}')
        all_lines.append((a,b,matches))

    # Write pairs file with all matches (one line per matched pair)
    with open(args.output_pairs, 'w') as f:
        for a,b,matches in all_lines:
            line = f"{os.path.basename(a)} {os.path.basename(b)} {len(matches)}"
            for i,j,s in matches:
                line += f" {i} {j}"
            f.write(line + "\n")

    print(f'Wrote pairs to {args.output_pairs}')


if __name__ == '__main__':
    main()
