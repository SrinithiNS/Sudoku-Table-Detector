Sudoku puzzles are popular recreational games that challenge logical reasoning and problem-solving skills. 
Extracting Sudoku puzzles from images efficiently and accurately is essential for their digitization and integration into digital platforms.
This project presents a framework, the Sudoku Table Extractor, designed to automatically extract Sudoku puzzles from images. 

Using computer vision techniques, the framework employs a multi-step approach including image preprocessing, grid detection, cell segmentation, and digit recognition.
The image preprocessing involves dilating and adaptive thresholding the image,following using floodfill algorithm to find the grid.
And using hough lines, the extreme lines are detected and using those lines, the grid is cropped out. 

Following that, the cells are divided equally, and after doing flood filling to reduce the noise, the digits are recognized using a CNN model.
