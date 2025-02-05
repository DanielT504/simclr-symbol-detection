Bobyard Computer Vision Challenge
===================================================
This project demonstrates fine-tuning for SimCLR using a firefighting device detection dataset from Roboflow.

===================================================
Installation & Setup
===================================================
1. Create and Activate a Virtual Environment:
   - e.g. Windows:
     python -m venv bobyard_env
     bobyard_env\Scripts\activate

2. Install Dependencies:
     pip install -r requirements.txt

===================================================
Running the Model
===================================================
Train the Model:
     python scripts/train.py
   - This will train for 5 epochs with data augmentation to prevent overfitting.
   - Model checkpoints are saved in models/.

Evaluate the Model:
     python scripts/evaluate.py
   - This loads the trained model and evaluates test accuracy.
   - Expected output is test accuracy + loss.

===================================================
Challenges Faced
===================================================
Overfitting Issue:
- During testing, the model achieves 100% accuracy, which is unrealistic.
- This suggests the model is memorizing the training data rather than generalizing.

===================================================
Steps Taken to Reduce Overfitting
===================================================
- Added Data Augmentation: Introduced random rotation, flipping, and color jitter.
- Lowered Learning Rate: Slowed down weight updates to improve generalization.
- Added Dropout: Introduced 0.5 dropout to reduce reliance on specific neurons.
- Introduced Validation Loss Tracking: Monitored overfitting between training and validation.

===================================================
Future Steps
===================================================
- Try a Smaller Model: ResNet-50 might be too complex for this dataset; a ResNet-18 model could generalize better.
- Introduce Label Smoothing: Prevents the model from becoming overly confident in predictions.
- Increase Training Data: If dataset size is small, synthetic augmentations could help.
- Use Different Loss Functions: Test other loss functions like Focal Loss to improve learning balance.
