# Neural Network Models for MNIST Classification

This repository contains four neural network models designed to classify MNIST digits. The goal of this assignment is to reach an accuracy of 99.40 % (or more in continous epochs) within 15 epochs and less than 8000 parameters

## Overview
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). The objective is to classify these images into the correct digit class. All models in this repository are implemented using PyTorch.

## Requirements
To run the models, ensure the following dependencies are installed:
- Python 3.7+
- PyTorch
- torchvision
- tqdm
- torchsummary

Install dependencies using:
```bash
pip install torch torchvision tqdm torchsummary
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run any model notebook:
   ```bash
   jupyter notebook Model1.ipynb
   ```
3. Follow the instructions in the notebook to train and test the model.

## Models

### Model 1
 **Logs**:
 ```python
  EPOCH: 0
  Loss=0.1615714132785797 Batch_id=468 Accuracy=94.83: 100%|██████████| 469/469 [00:17<00:00, 27.00it/s]

  Test set: Average loss: 0.0809, Accuracy: 9874/10000 (98.74%)

  EPOCH: 1
  Loss=0.09870464354753494 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:17<00:00, 27.16it/s]

  Test set: Average loss: 0.0598, Accuracy: 9891/10000 (98.91%)

  EPOCH: 2
  Loss=0.06682561337947845 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:16<00:00, 27.78it/s]

  Test set: Average loss: 0.0463, Accuracy: 9896/10000 (98.96%)

  EPOCH: 3
  Loss=0.03591493144631386 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:17<00:00, 27.19it/s]

  Test set: Average loss: 0.0398, Accuracy: 9908/10000 (99.08%)

  EPOCH: 4
  Loss=0.01757017709314823 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:16<00:00, 27.86it/s]

  Test set: Average loss: 0.0362, Accuracy: 9911/10000 (99.11%)

  EPOCH: 5
  Loss=0.021575862541794777 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:16<00:00, 28.43it/s]

  Test set: Average loss: 0.0314, Accuracy: 9923/10000 (99.23%)

  EPOCH: 6
  Loss=0.023224657401442528 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:16<00:00, 27.80it/s]

  Test set: Average loss: 0.0316, Accuracy: 9925/10000 (99.25%)

  EPOCH: 7
  Loss=0.023533277213573456 Batch_id=468 Accuracy=99.27: 100%|██████████| 469/469 [00:16<00:00, 28.22it/s]

  Test set: Average loss: 0.0260, Accuracy: 9935/10000 (99.35%)

  EPOCH: 8
  Loss=0.008406040258705616 Batch_id=468 Accuracy=99.30: 100%|██████████| 469/469 [00:16<00:00, 28.28it/s]

  Test set: Average loss: 0.0306, Accuracy: 9915/10000 (99.15%)

  EPOCH: 9
  Loss=0.022673368453979492 Batch_id=468 Accuracy=99.37: 100%|██████████| 469/469 [00:17<00:00, 26.69it/s]

  Test set: Average loss: 0.0269, Accuracy: 9932/10000 (99.32%)

  EPOCH: 10
  Loss=0.02085181511938572 Batch_id=468 Accuracy=99.38: 100%|██████████| 469/469 [00:17<00:00, 26.67it/s]

  Test set: Average loss: 0.0220, Accuracy: 9946/10000 (99.46%)

  EPOCH: 11
  Loss=0.02772708795964718 Batch_id=468 Accuracy=99.49: 100%|██████████| 469/469 [00:20<00:00, 23.41it/s]

  Test set: Average loss: 0.0260, Accuracy: 9930/10000 (99.30%)

  EPOCH: 12
  Loss=0.029824933037161827 Batch_id=468 Accuracy=99.46: 100%|██████████| 469/469 [00:16<00:00, 28.18it/s]

  Test set: Average loss: 0.0248, Accuracy: 9927/10000 (99.27%)

  EPOCH: 13
  Loss=0.04310476407408714 Batch_id=468 Accuracy=99.48: 100%|██████████| 469/469 [00:16<00:00, 28.75it/s]

  Test set: Average loss: 0.0225, Accuracy: 9939/10000 (99.39%)

  EPOCH: 14
  Loss=0.017283542081713676 Batch_id=468 Accuracy=99.51: 100%|██████████| 469/469 [00:17<00:00, 27.07it/s]

  Test set: Average loss: 0.0228, Accuracy: 9931/10000 (99.31%)
```

**TARGET:**

Make the model lighter

Keep the epochs within 15

Keep the parameters less than 8000
    
**RESULT:**

Parameters : 9,954

Best Training Accuracy : 99.51

Best Test Accuracy : 99.46

**ANALYSIS:**

Total number of parameters is not less than 8000

The total number of epochs used is 15

The model was good till 1th epoch but from the 11th epoch we could see the model is overfitting

**WHAT CAN BE DONE:**

Change the convolution layers to keep the parameters in check

Introduce dropout to reduce over fitting

### Model 2

**Logs**:
 ```python
 EPOCH: 0
Loss=0.12093589454889297 Batch_id=468 Accuracy=94.22: 100%|██████████| 469/469 [00:16<00:00, 28.40it/s]

Test set: Average loss: 0.0962, Accuracy: 9843/10000 (98.43%)

EPOCH: 1
Loss=0.11352702230215073 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:16<00:00, 28.77it/s]

Test set: Average loss: 0.0601, Accuracy: 9882/10000 (98.82%)

EPOCH: 2
Loss=0.08246047049760818 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:16<00:00, 27.76it/s]

Test set: Average loss: 0.0479, Accuracy: 9905/10000 (99.05%)

EPOCH: 3
Loss=0.030631719157099724 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:17<00:00, 27.35it/s]

Test set: Average loss: 0.0388, Accuracy: 9916/10000 (99.16%)

EPOCH: 4
Loss=0.04610821604728699 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:16<00:00, 28.51it/s]

Test set: Average loss: 0.0380, Accuracy: 9915/10000 (99.15%)

EPOCH: 5
Loss=0.029482951387763023 Batch_id=468 Accuracy=99.09: 100%|██████████| 469/469 [00:16<00:00, 28.84it/s]

Test set: Average loss: 0.0341, Accuracy: 9924/10000 (99.24%)

EPOCH: 6
Loss=0.047235745936632156 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:17<00:00, 26.53it/s]

Test set: Average loss: 0.0297, Accuracy: 9932/10000 (99.32%)

EPOCH: 7
Loss=0.025506870821118355 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:16<00:00, 28.67it/s]

Test set: Average loss: 0.0276, Accuracy: 9931/10000 (99.31%)

EPOCH: 8
Loss=0.02761770784854889 Batch_id=468 Accuracy=99.26: 100%|██████████| 469/469 [00:16<00:00, 28.63it/s]

Test set: Average loss: 0.0279, Accuracy: 9923/10000 (99.23%)

EPOCH: 9
Loss=0.008381711319088936 Batch_id=468 Accuracy=99.29: 100%|██████████| 469/469 [00:17<00:00, 27.47it/s]

Test set: Average loss: 0.0263, Accuracy: 9927/10000 (99.27%)

EPOCH: 10
Loss=0.010238707065582275 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:16<00:00, 28.73it/s]

Test set: Average loss: 0.0260, Accuracy: 9935/10000 (99.35%)

EPOCH: 11
Loss=0.05624288693070412 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:16<00:00, 29.14it/s]

Test set: Average loss: 0.0247, Accuracy: 9938/10000 (99.38%)

EPOCH: 12
Loss=0.012332934886217117 Batch_id=468 Accuracy=99.38: 100%|██████████| 469/469 [00:17<00:00, 26.91it/s]

Test set: Average loss: 0.0243, Accuracy: 9927/10000 (99.27%)

EPOCH: 13
Loss=0.052587468177080154 Batch_id=468 Accuracy=99.42: 100%|██████████| 469/469 [00:16<00:00, 28.52it/s]

Test set: Average loss: 0.0306, Accuracy: 9912/10000 (99.12%)

EPOCH: 14
Loss=0.010730399750173092 Batch_id=468 Accuracy=99.40: 100%|██████████| 469/469 [00:16<00:00, 28.98it/s]

Test set: Average loss: 0.0251, Accuracy: 9935/10000 (99.35%)
 ```

**TARGET**

Keep the parameters less than 8000

Reduce the overfitting

Reach test accuracy > 99.4

**RESULT**

Parameters : 8,292

Best Training Accuracy : 99.42

Best Test Accuracy : 99.38

**ANALYSIS**

Parameters count is still greater than 8000

We could see model overfitting from 12th epoch

Even if we continue for few more epochs the model will not reach test accuracy > 99.4

**WHAT CAN BE DONE**

Reduce the parameter count by reduce the number of feature maps in intermediate layers

Add image augmentation

Increase the drop out

### Model 3
**Logs**:
 ```python
EPOCH: 0
Loss=0.11774060130119324 Batch_id=468 Accuracy=93.16: 100%|██████████| 469/469 [00:32<00:00, 14.48it/s]

Test set: Average loss: 0.0979, Accuracy: 9820/10000 (98.20%)

EPOCH: 1
Loss=0.04535025358200073 Batch_id=468 Accuracy=97.93: 100%|██████████| 469/469 [00:31<00:00, 15.04it/s]

Test set: Average loss: 0.0611, Accuracy: 9875/10000 (98.75%)

EPOCH: 2
Loss=0.09495209902524948 Batch_id=468 Accuracy=98.35: 100%|██████████| 469/469 [00:32<00:00, 14.57it/s]

Test set: Average loss: 0.0555, Accuracy: 9862/10000 (98.62%)

EPOCH: 3
Loss=0.07882190495729446 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:31<00:00, 15.07it/s]

Test set: Average loss: 0.0390, Accuracy: 9909/10000 (99.09%)

EPOCH: 4
Loss=0.06368156522512436 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:31<00:00, 15.06it/s]

Test set: Average loss: 0.0344, Accuracy: 9914/10000 (99.14%)

EPOCH: 5
Loss=0.04764308035373688 Batch_id=468 Accuracy=98.70: 100%|██████████| 469/469 [00:31<00:00, 15.11it/s]

Test set: Average loss: 0.0357, Accuracy: 9906/10000 (99.06%)

EPOCH: 6
Loss=0.10236049443483353 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:30<00:00, 15.15it/s]

Test set: Average loss: 0.0361, Accuracy: 9898/10000 (98.98%)

EPOCH: 7
Loss=0.03791040554642677 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:30<00:00, 15.26it/s]

Test set: Average loss: 0.0311, Accuracy: 9922/10000 (99.22%)

EPOCH: 8
Loss=0.09331950545310974 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:31<00:00, 14.66it/s]

Test set: Average loss: 0.0290, Accuracy: 9913/10000 (99.13%)

EPOCH: 9
Loss=0.05725391209125519 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:30<00:00, 15.33it/s]

Test set: Average loss: 0.0259, Accuracy: 9932/10000 (99.32%)

EPOCH: 10
Loss=0.02974691428244114 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:30<00:00, 15.28it/s]

Test set: Average loss: 0.0274, Accuracy: 9923/10000 (99.23%)

EPOCH: 11
Loss=0.053218577057123184 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:31<00:00, 14.81it/s]

Test set: Average loss: 0.0242, Accuracy: 9931/10000 (99.31%)

EPOCH: 12
Loss=0.02607146091759205 Batch_id=468 Accuracy=99.10: 100%|██████████| 469/469 [00:30<00:00, 15.32it/s]

Test set: Average loss: 0.0242, Accuracy: 9930/10000 (99.30%)

EPOCH: 13
Loss=0.023845404386520386 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:31<00:00, 14.82it/s]

Test set: Average loss: 0.0230, Accuracy: 9932/10000 (99.32%)

EPOCH: 14
Loss=0.03304101154208183 Batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:30<00:00, 15.38it/s]

Test set: Average loss: 0.0236, Accuracy: 9927/10000 (99.27%)
```

**TARGET**

Keep the parameters less than 8000

Reduce the overfitting

Reach test accuracy > 99.4

**RESULT**

Parameters :  6,798

Best Training Accuracy : 99.15

Best Test Accuracy : 99.32

**ANALYSIS**

Total parameters has been kept under 8000

The model is not overfitting any more

Required test accuracy > 99.4% has not been reached yet

**WHAT CAN BE DONE**

Gradually increase the number of feature maps in intermediate layers but still keep the count under 8000

Introducing scheduler like StepLR to adjust the learning rate during training

Trying Adam optimizer with lower learning rate for optimization

### Model 4 - Trained on EC2
**Logs**:
 ```python
EPOCH: 0
Loss=0.2490364909172058 Batch_id=468 Accuracy=93.12: 100%|██████████| 469/469 [00:10<00:00, 44.72it/s] 

Test set: Average loss: 0.2375, Accuracy: 9816/10000 (98.16%)

EPOCH: 1
Loss=0.15466056764125824 Batch_id=468 Accuracy=98.00: 100%|██████████| 469/469 [00:07<00:00, 66.28it/s]

Test set: Average loss: 0.1369, Accuracy: 9848/10000 (98.48%)

EPOCH: 2
Loss=0.16827517747879028 Batch_id=468 Accuracy=98.43: 100%|██████████| 469/469 [00:07<00:00, 66.85it/s] 

Test set: Average loss: 0.0837, Accuracy: 9880/10000 (98.80%)

EPOCH: 3
Loss=0.09689375758171082 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:07<00:00, 66.48it/s] 

Test set: Average loss: 0.0666, Accuracy: 9890/10000 (98.90%)

EPOCH: 4
Loss=0.10837340354919434 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:07<00:00, 65.74it/s] 

Test set: Average loss: 0.0511, Accuracy: 9912/10000 (99.12%)

EPOCH: 5
Loss=0.06592915207147598 Batch_id=468 Accuracy=98.88: 100%|██████████| 469/469 [00:07<00:00, 66.26it/s] 

Test set: Average loss: 0.0471, Accuracy: 9908/10000 (99.08%)

EPOCH: 6
Loss=0.10578694194555283 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:07<00:00, 66.98it/s] 

Test set: Average loss: 0.0488, Accuracy: 9902/10000 (99.02%)

EPOCH: 7
Loss=0.03217274695634842 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:06<00:00, 67.42it/s] 

Test set: Average loss: 0.0386, Accuracy: 9921/10000 (99.21%)

EPOCH: 8
Loss=0.024886390194296837 Batch_id=468 Accuracy=99.27: 100%|██████████| 469/469 [00:06<00:00, 68.01it/s]

Test set: Average loss: 0.0269, Accuracy: 9943/10000 (99.43%)

EPOCH: 9
Loss=0.050524670630693436 Batch_id=468 Accuracy=99.28: 100%|██████████| 469/469 [00:06<00:00, 67.43it/s]

Test set: Average loss: 0.0273, Accuracy: 9938/10000 (99.38%)

EPOCH: 10
Loss=0.027504606172442436 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:07<00:00, 66.31it/s]

Test set: Average loss: 0.0258, Accuracy: 9940/10000 (99.40%)

EPOCH: 11
Loss=0.034093741327524185 Batch_id=468 Accuracy=99.31: 100%|██████████| 469/469 [00:07<00:00, 66.67it/s]

Test set: Average loss: 0.0257, Accuracy: 9946/10000 (99.46%)

EPOCH: 12
Loss=0.024683421477675438 Batch_id=468 Accuracy=99.38: 100%|██████████| 469/469 [00:07<00:00, 66.29it/s]

Test set: Average loss: 0.0249, Accuracy: 9948/10000 (99.48%)

EPOCH: 13
Loss=0.08600360155105591 Batch_id=468 Accuracy=99.36: 100%|██████████| 469/469 [00:06<00:00, 67.06it/s] 

Test set: Average loss: 0.0247, Accuracy: 9948/10000 (99.48%)

EPOCH: 14
Loss=0.05660015344619751 Batch_id=468 Accuracy=99.39: 100%|██████████| 469/469 [00:06<00:00, 67.24it/s] 

Test set: Average loss: 0.0244, Accuracy: 9947/10000 (99.47%)
```
**TARGET**:

Keep the total parameter count less than 8000

Reach test accuracy > 99.4 in less than 15 epochs

**RESULT**:

Parameters : 7,530

Best Training Accuracy : 99.39

Best Test Accuracy : 99.48

**ANALYSIS**:

Model is underfitting

Test accuracy has reached 99.4 and is consistent from 10th epoch

Adding StepLR helped the model to perform better

**EC2_LOG_IMAGES**

Refer Model4_EC2_0 to 4.png

Refer Model4_EC2_5 to 9.png

Refer Model4_EC2_10 to 14.png

## Evaluation

- **Optimizer**:  SGD with momentum for first 3 models and Adam for the 4th model
- **Loss Function**: Negative Log Likelihood (NLL Loss).
- **Epochs**: 15
- **Batch Size**: 128 (when CUDA is available).

