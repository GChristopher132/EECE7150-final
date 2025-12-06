SuperGlue prototype example

This folder contains a prototype script `superglue_matcher.py` that demonstrates
how to run SuperGlue on a pair of images and import matches into COLMAP.

Requirements (install into your Python environment):

- python >= 3.8
- numpy
- torch (if using SuperGlue PyTorch models)
- pycolmap (compiled extension)
- opencv-python (for OpenCV SIFT fallback)

Usage (example):

python -m pycolmap.examples.superglue_matcher \
  --image0 /path/to/img0.jpg \
  --image1 /path/to/img1.jpg \
  --superglue_dir /path/to/SuperGluePretrainedNetwork \
  --output_pairs pairs.txt

Notes:
- The script contains heuristics to import common SuperGlue/SuperPoint repo layouts, but
  you may need to adapt the `load_superglue_module` and `run_superglue_on_pair` functions
  to your local implementation.
- The script writes a simple pairwise match text file which can be imported into a
  COLMAP database or used with COLMAP's `matches_importer`.
