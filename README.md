# YOLO + MobileNetV2 for Aggression Detection in 12 Videos with 10 sec per vidoes as dataset with 84% accuracy

This project combines **YOLOv8** for person detection with **MobileNetV2** for aggression classification 
(`aggressive` vs. `non_aggressive`) in video frames.  
The pipeline takes videos as input, extracts frames, applies preprocessing and augmentation, 
then trains a classifier head on top of MobileNetV2.  
Finally, it runs inference, draws bounding boxes and labels on frames, and generates an output video.

---

## 📌 Features
- Uses **YOLOv8x** for detecting persons in frames.
- Uses **MobileNetV2 (transfer learning)** for binary classification.
- Training with **CrossEntropyLoss** + Adam optimizer + EarlyStopping.
- Saves the best model based on validation loss.
- Outputs **classification report** (precision, recall, F1).
- Generates an annotated **video output** with predictions.

---

## 📂 Dataset
- Structure expected:
