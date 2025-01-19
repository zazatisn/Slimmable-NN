# Slimmable Dynamic NN

This repo contains the scripts and models used for the Slimmable variance of Dynamic NNs. 

The models that are currently supported are a smaller version of:
- ResNet9
- AlexNet

**NOTE:** Those models can be easily expanded for better accuracy

## Used / Tested Libraries
- Torch: 2.5.1
- Numpy: 1.23.5
- Neural Compressor: 2.6
- Pandas: 2.2.1

## Train Scripts
- train_alexnet.py: Trains AlexNet and saves model

  Syntax: `python3 train_alexnet.py (cifar10|cifar100) <batch_size>`
  
- train_resnet_v2.py: Trains ResNet9 and saves model

  Syntax: `python3 train_resnet_v2.py (cifar10|cifar100) <batch_size>`

- train_slimmable_alexnet.py: Train Slimmable AlexNet and saves each increment

  Syntax: `python3 train_slimmable_alexnet.py <batch_size> <num_increments> <subnetwork_ratio> (cifar10|cifar100)`

- train_slimmable_resnet9_v2.py: Train Slimmable ResNet9 and saves each increment

  Syntax: `python3 train_resnet9 <batch_size> <num_increments> <subnetwork_ratio> (cifar10|cifar100)`

## Quantize Script
- quantize_model.py: Quantize trained fp32 model to int8 and saves the quantized model
  
  **NOTE:** Some torch layers are not supported while others may not be quantized to int8 

  Syntax: `python3 quantize_model.py <fp_model_foldername> <quantized_model_foldername> (cifar10|cifar100)`

## Evaluate Scripts
- evaluate_net.py: Evaluates loss, accuracy for train and test dataset for a given model. If model is slimmable net then evaluates each increment seperately

  Syntax: `python3 evaluate_net.py <model_foldername> (cifar10|cifar100)`

- evaluate_net.py: Evaluates loss, accuracy for train and test dataset for a given quantized model. If quantized model is slimmable net then evaluates each quantized increment seperately

  Syntax: `python3 quantize_model.py <fp_model_foldername> <quantized_model_foldername> (cifar10|cifar100)`

- evaluate_dynamic_nn.py: Evaluates accuracy and average increments for a given slimmable net using different threshold/margins. Stores the statistics to csv files for easy plotting using **plot_dynamic_nn_metric.py**

  Syntax: `python3 evaluate_dynamic_nn.py <model_foldername> <csv_filename> (cifar10|cifar100)` 