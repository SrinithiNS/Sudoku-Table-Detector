# Sudoku Table Extractor

This repository contains a computer-vision pipeline for detecting, extracting, and digitizing 9x9 Sudoku puzzles from photographs or scanned images. The goal is to reliably locate the Sudoku grid in a heterogeneous scene, segment it into individual cells, and produce a clean, 9x9 numeric matrix suitable for downstream solvers.

## Key features

- Robust grid detection using contour analysis and Hough line detection
- Perspective correction (homography) to obtain a top-down view of the puzzle
- Precise 9x9 cell segmentation and per-cell noise reduction
- Digit classification using a small CNN (custom-trained or pre-trained model)
- Simple inference pipeline that outputs a 9x9 integer matrix with 0 for empty cells

## Pipeline / Algorithmic details

1. Image preprocessing
   - Convert to grayscale and apply Gaussian blur to reduce camera noise.
   - Apply adaptive thresholding (Gaussian or mean) to binarize under uneven lighting.
   - Use morphological operations (dilate / erode / opening) to strengthen grid lines and remove small noise.

2. Grid detection
   - Find external contours and select the largest quadrilateral-like contour by area and approximate polygonal curves (cv2.approxPolyDP).
   - If contour-based detection fails (occlusion, severe perspective), fall back to probabilistic Hough Line Transform to detect long straight lines.
   - From Hough lines, estimate the four boundary lines (top, bottom, left, right) using clustering of line angles and distances.
   - Compute intersection points of the boundary lines to form a quadrilateral and use cv2.getPerspectiveTransform / cv2.warpPerspective to rectify the puzzle to a square top-down view.

3. Grid refinement
   - On the rectified top-down image, optionally re-run adaptive thresholding and morphological filtering to obtain clean, orthogonal grid lines.
   - Use horizontal and vertical projection profiles or Hough lines to refine line locations.
   - Compute exact cell boundaries by splitting the rectangular grid into 9 equal rows and 9 equal columns (floating-point arithmetic + rounding to pixel boundaries).

4. Cell extraction and cleaning
   - Crop each cell individually with a small inner margin to avoid boundary artifacts.
   - Apply local noise reduction: flood-fill to remove spurious small components, morphological opening to remove salt-and-pepper noise, and contour filtering by area to preserve digit-like blobs.
   - Normalize each cell to a fixed size (e.g., 28x28 or 32x32) and center the digit by computing the digit bounding box and centering the mass.

5. Digit recognition
   - Feed normalized cell images into a small convolutional neural network (CNN) classifier that outputs probability distribution over 0-9 (0 indicates empty). Example architecture:
     - Input: 28x28 grayscale
     - Conv 32 (3x3) -> ReLU -> MaxPool
     - Conv 64 (3x3) -> ReLU -> MaxPool
     - Dense 128 -> Dropout
     - Softmax output (10 classes)
   - Use cross-entropy loss and standard data augmentation (rotation +/- 10°, scaling, translation, brightness) during training to improve generalization.
   - Optionally fine-tune on a labeled dataset of extracted digits from real Sudoku photos (synthetic + manual labels helps).

6. Post-processing
   - Interpret low-confidence predictions as empty cells if predicted probability < threshold (e.g., 0.6) or if the cell has very few foreground pixels.
   - Validate the resulting 9x9 matrix for basic consistency (digits 1–9, optional quick Sudoku solver to check solvability).

## Evaluation and metrics

- Per-cell classification accuracy: percentage of cells correctly classified (including correctly detected empties).
- Puzzle extraction success rate: fraction of input images where the pipeline returns a valid 9x9 matrix (no missing cells and reasonable digit detection).
- End-to-end puzzle solve rate: fraction of extracted puzzles that a Sudoku solver can solve to completion.

When reporting numbers, include dataset details (number of images, source, augmentation strategy) and evaluation protocol (train/val/test split, cross-validation if used).

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- TensorFlow / Keras or PyTorch (depending on the model implementation)
- scikit-image (optional, for morphology utilities)

Install via pip:

pip install -r requirements.txt

(If you do not have a requirements file, install: opencv-python numpy tensorflow scikit-image)

## Usage

1. Place an input image (photo of a Sudoku) in an accessible path.
2. Run the inference script (example):

python infer.py --image path/to/sudoku.jpg --model models/digit_cnn.h5 --output results.json

Output format: JSON with a 9x9 array and per-cell confidence scores. Example:

{
  "matrix": [[5,3,0,0,7,0,0,0,0], ...],
  "confidences": [[0.98, 0.95, 0.12, ...], ...]
}

Include flags to visualize intermediate steps (preprocessed image, detected grid, per-cell crops) for debugging.

## Model and dataset

- The repository expects a trained digit classifier at models/digit_cnn.h5 (or a PyTorch .pt). Provide instructions or a training script to reproduce the model.
- Recommended training data: MNIST + synthetically generated digits rendered on grid backgrounds + hand-labeled crops from real photos. Use aggressive augmentation (rotation, scale, elastic transforms) to cover camera variability.

## Known limitations and failure modes

- Very low contrast photos or extreme perspective/occlusion may prevent accurate grid detection.
- Heavy shadows, reflections, or handwriting styles not covered during training may lower digit recognition accuracy.
- Assumes classical printed digits; highly stylized or overlapping digits may be misclassified.

## Contributing

Contributions are welcome. Suggested improvements:
- Add a robust fallback for missing grid corners (use Hough-based line clustering).
- Integrate a CRNN / sequence model for merged digits or multi-digit cells.
- Provide a labeled dataset of real photos and a reproducible training pipeline.

Please open issues or pull requests against this repository.

## License

Specify repository license here (e.g., MIT). If none, add a LICENSE file.
