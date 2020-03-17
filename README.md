# danceGANs

## Dependencies

For running this code and reproducing the results, you need the following packages. Python 2.7 and 3.6 has been used.

Packages:

- TensorFlow 1.0 and 2.0
- NumPy
- cv2
- scikit-video
- scikit-image
- Glob

## How to Run

Place the videos inside a folder called "trainvideos".
Run main.py with the required values for each flag variable (set in main.py).

If the number of frames/video-size flag is changed, modify it directly in the parameter. Then modify the read-and-process-videos function in utils.py to pick the correct number of frames from the read video.

## Project Structure

danceGANs_v1, tf_updated_v2 - Python 2, Tensorflow 1
Everything else - Python 3, Tensorflow 2
