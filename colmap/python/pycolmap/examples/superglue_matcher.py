"""
Simple pycolmap-based SuperGlue matcher prototype.

Requirements:
 - pycolmap must be importable (the compiled _core extension)
 - A SuperGlue + SuperPoint Python implementation available at
   EECE7150/HW4/SuperGluePretrainedNetwork (or update path)
 - PyTorch + GPU recommended for speed

This script demonstrates the data flow and writes matches to a COLMAP-style pair text file.

Usage:
  python -m pycolmap.examples.superglue_matcher \
      --image0 path/to/img0.jpg --image1 path/to/img1.jpg \
      --output_pairs pairs.txt --superglue_dir /path/to/SuperGluePretrainedNetwork

"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    import pycolmap
except Exception as e:
    raise RuntimeError("pycolmap import failed; ensure pycolmap is installed") from e

# Try to import SuperGlue code by adding its path


def load_superglue_module(superglue_dir: str):
    """Add superglue dir to path and import the model helper.

    This expects the SuperGlue repo layout used in many public forks where a
    `models` or `superpoint` module exists. You may need to adapt imports.
    """
    sys.path.insert(0, superglue_dir)
    try:
        # common entrypoints in popular repos
        import models
        return models
    except Exception:
        try:
            import superpoint
            import superglue
            return (superpoint, superglue)
        except Exception as ex:
            raise RuntimeError(
                "Could not import SuperGlue modules from path: %s" % superglue_dir
            ) from ex


def extract_keypoints_with_pycolmap(image_path: str):
    """Extract SIFT keypoints & descriptors using pycolmap's extractor.

    Returns:
      keypoints: (N,2) float32
      descriptors: (N,128) float32
    """
    # Use pycolmap extractors if available. If not, fallback to loading from file.
    # This is a minimal abstraction; adjust for your pycolmap build.
    try:
        image = pycolmap.Image(image_path)
    except Exception:
        # pycolmap may not provide a direct Image wrapper for loading; instead use
        # the feature extractor APIs. For the prototype, we fall back to OpenCV SIFT
        import cv2

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        return pts, desc.astype(np.float32)

    # If pycolmap has extract features API: use pycolmap.extract_features
    try:
        feats = pycolmap.extract_features(image_path)
        # feats likely contains keypoints and descriptors
        kps = np.array([[k.x, k.y] for k in feats.keypoints], dtype=np.float32)
        desc = np.array(feats.descriptors, dtype=np.float32)
        return kps, desc
    except Exception:
        raise RuntimeError("pycolmap feature extraction not available in this build")


def run_superglue_on_pair(
    kps0: np.ndarray,
    desc0: np.ndarray,
    kps1: np.ndarray,
    desc1: np.ndarray,
    superglue_mod,
    device: str = "cuda",
) -> List[Tuple[int, int, float]]:
    """Run SuperGlue given keypoints/descriptors from two images.

    Returns a list of (idx0, idx1, score).
    This wrapper assumes the imported superglue module provides a callable API
    similar to the public implementations. You may need to adapt to your
    local repo's API.
    """
    # The exact calling convention varies. We try a few heuristics.
    # Standard public API expects dicts {"keypoints0":..., "descriptors0":...}
    if isinstance(superglue_mod, tuple):
        # (superpoint, superglue) import style
        superpoint, superglue = superglue_mod
        # Create networks and run inference - placeholder; user must adapt
        raise RuntimeError("Detected modular import for SuperPoint/SuperGlue;\n"
                           "Please adapt this script to construct the models from your repo.")

    # If `models` module is present and contains SuperGlue model builder:
    if hasattr(superglue_mod, "SuperGlue"):
        # instantiate model with default params
        Model = superglue_mod.SuperGlue
        cfg = {k: getattr(superglue_mod, k) for k in dir(superglue_mod) if k.isupper()}
        model = Model(cfg).to(device)
        model.eval()

        # Prepare input tensors
        import torch

        kps0_t = torch.from_numpy(kps0)[None].float().to(device)
        kps1_t = torch.from_numpy(kps1)[None].float().to(device)
        desc0_t = torch.from_numpy(desc0)[None].float().to(device)
        desc1_t = torch.from_numpy(desc1)[None].float().to(device)

        with torch.no_grad():
            # many implementations expect normalized keypoints (0..W,0..H) -> normalized -1..1
            out = model({'keypoints0': kps0_t, 'descriptors0': desc0_t,
                         'keypoints1': kps1_t, 'descriptors1': desc1_t})
        matches0 = out['matches0'][0].cpu().numpy()
        match_scores0 = out.get('match_scores0')
        if match_scores0 is not None:
            match_scores0 = match_scores0[0].cpu().numpy()
        matches = []
        for i, j in enumerate(matches0):
            if j >= 0:
                score = float(match_scores0[i]) if match_scores0 is not None else 1.0
                matches.append((int(i), int(j), score))
        return matches

    raise RuntimeError("Unsupported SuperGlue module layout; adapt script to your local repo")


def write_pair_matches_text(path: str, image0: str, image1: str, matches: List[Tuple[int, int, float]]):
    """Write match list to a simple pair text file consumable by COLMAP importers.

    Format (simple text): one line per pair:
      image0 image1 num_matches i0 j0 i1 j1 ...

    Where iN/jN are keypoint indices (0-based).
    """
    with open(path, "w") as f:
        # header: "# pairs"
        line = f"{os.path.basename(image0)} {os.path.basename(image1)} {len(matches)}"
        for i, j, s in matches:
            line += f" {i} {j}"
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--superglue_dir", required=True,
                        help="Path to your SuperGluePretrainedNetwork checkout")
    parser.add_argument("--output_pairs", default="pairs.txt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    kps0, desc0 = extract_keypoints_with_pycolmap(args.image0)
    kps1, desc1 = extract_keypoints_with_pycolmap(args.image1)
    print(f"Extracted {len(kps0)} / {len(kps1)} keypoints")

    sg_mod = load_superglue_module(args.superglue_dir)
    matches = run_superglue_on_pair(kps0, desc0, kps1, desc1, sg_mod, device=args.device)
    print(f"Found {len(matches)} matches from SuperGlue")

    write_pair_matches_text(args.output_pairs, args.image0, args.image1, matches)
    print(f"Wrote pairs to {args.output_pairs}")


if __name__ == "__main__":
    main()
