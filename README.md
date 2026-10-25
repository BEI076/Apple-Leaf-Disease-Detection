# Apple Leaf Disease Detection

Upload a photo of an apple leaf, and a CNN model tells you if it is healthy or has one of three diseases. It also suggests a treatment.

A BEI076 minor project (2023), built with TensorFlow/Keras and Streamlit.

## What it detects

| Class | Suggested treatment |
|---|---|
| Apple scab | Fungicides; prune and thin the trees so air can move |
| Black rot | Fungicides; prune to let in air and light |
| Cedar apple rust | Fungicides; prune infected branches; remove badly infected cedar trees nearby |
| Healthy | No treatment needed |

## How it works

1. You sign up or log in (Firebase Authentication).
2. You upload a JPG or PNG photo.
3. The app checks that the photo is a leaf: at least 5% of the pixels must be green.
4. The image is resized to 256 × 256 and passed to the model, `balanced_cnn_model.h5`.
5. The app shows the predicted class and how sure the model is. "Treatment" shows the advice above.

## Results

The model was trained for 20 epochs on a **balanced** dataset, with about 2,000 images in each class.

**Test accuracy: 97.2%** (test loss 0.097)

| Before balancing | After balancing |
|---|---|
| ![Class sizes before balancing](before%20bar.png) | ![Class sizes after balancing](bargraph%20after.png) |

![Loss and accuracy during training](lossvs%20acc.png)

## Run it

You need Python 3.10 and your own Firebase project (with Email/Password sign-in turned on and a Realtime Database).

1. Install the packages:

   ```sh
   pip install streamlit streamlit-option-menu streamlit-extras tensorflow opencv-python pillow matplotlib pyrebase4
   ```

   `requirements.txt` has the exact versions we used. It was made on Windows, so `tensorflow-intel` only installs there.

2. In `main.py`:
   - put your Firebase web settings in `firebaseconfig`
   - change the model path in `load_model(...)` to `"balanced_cnn_model.h5"` (it still points to a folder on the original laptop)

3. Start it:

   ```sh
   streamlit run main.py
   ```

## Files

| File | What it is |
|---|---|
| `main.py` | The Streamlit app |
| `balanced_cnn_model.h5` | The trained model (about 9.4 MB) |
| `*.png`, `Confusion matrix.docx` | Charts from training |
| `welcome.png`, `welcome2.png` | Pictures shown in the app |
