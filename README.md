# COLMAP using ML Feature Detection and Matching
To run COLMAP with these new capabilities, it takes a couple of extra steps

### 1. Activate the Python Virtual Environment
To do this, run the following from the root directory:
```bash
source /venv/bin/acivate
```

### 2. Run COLMAP from the build folder
Use the following command:
```bash
./colmap/build/src/colmap/exe/colmap gui
```

### 3. Make a new project
Follow the same instructions to do this as you would for any other COLMAP project

### 4. Feature Extraction
- Click the `Processing` tab, then `Feature Extraction`
- Select the `SuperPoint (ML)` tab
- Adjust any parameters and press extract. Wait for the python script to execute fully

### 5. Feature Matching
- Click the `Processing` tab, then `Feature Matching`
- Select `Type` and choose the ML feature matching you would like to use
- Press run and wait for the python script to execute fully

### 6. Start the Reconstruction
- Press the blue play arrow on the tool bar
- Wait for the reconstruction to complete

## Done!
