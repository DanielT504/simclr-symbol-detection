This project demonstrates fine-tuning for SimCLR using a firefighting device detection dataset from Roboflow.
=============================================================================================================

Installation & Setup
====================

1. Create and Activate a Virtual Environment:

   - e.g. Windows:
     python -m venv bobyard_env
     bobyard_env\Scripts\activate
2. Install Dependencies:
   pip install -r requirements.txt


Running the Model
=================

Train the Model:
     python scripts/train.py

- This will train for 5 epochs with data augmentation to prevent overfitting.
- Model checkpoints are saved in models/.

Evaluate the Model:
     python scripts/evaluate.py

- This loads the trained model and evaluates test accuracy.
- Expected output is test accuracy + loss.


Challenges Faced
================

Class Imbalance
     - Class distribution was highly imbalanced, with some classes having over 900 samples while others had less than 10.
     - Some classes had no samples in the test set.



Steps Taken to Improve Generalization
=================================

Implemented Class Balancing:
     - Used weighted loss function to compensate for imbalanced classes.


Future Steps
============

Increase Training Data
     - Many classes have very few training samples (some have fewer than 10).
     - More diverse examples or data augmentation for rare classes could improve performance.
Improve Test Set Diversity
     - The test set lacks examples from many classes.
     - Samples could be taken from the training set
