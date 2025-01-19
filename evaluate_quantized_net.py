import os
import sys
import utility_functions
import torch
import alexnet
import resnet9_v2
from neural_compressor.utils.pytorch import load

syntax = "python3 evaluate_quantized_net.py <model_foldername> <quantized_model_foldername> (cifar10|cifar100)"

# store networks in dictionary #
nets = {}
quantized_nets = {}

if len(sys.argv) != 4:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)
else:
  model_foldername = sys.argv[1]
  quantized_model_foldername = sys.argv[2]
  if not os.path.exists(model_foldername):
    print(f"ERROR: Folder {model_foldername} does not exist!")
    print(f"Rerun: {syntax}")
    sys.exit(-2)
  elif not os.path.exists(quantized_model_foldername):
    print(f"ERROR: Folder {quantized_model_foldername} does not exist!")
    print(f"Rerun: {syntax}")
    sys.exit(-2)
  else:
    dir_list = os.listdir(model_foldername)
    for file in dir_list:
      print(file)
      full_model_filepath = os.path.join(model_foldername, file)
      full_quantized_model_filepath = os.path.join(quantized_model_foldername, file)
      nets[file[:-3]] = torch.load(full_model_filepath)
      quantized_nets[file[:-3]] = load(full_quantized_model_filepath, nets[file[:-3]])
    
  if sys.argv[3].lower() != "cifar10" and sys.argv[3].lower() != "cifar100":
    print(f"ERROR: Invalid dataset {sys.argv[2]}")
    print(f"Rerun: {syntax}")
    sys.exit(-1)

# get dataset in Dataloader #
dataset = sys.argv[3].lower()
if dataset == "cifar10":
  trainloader, testloader = utility_functions.load_cifar10(batch_size=200, num_workers=2)
  num_classes = 10
elif dataset == "cifar100":
  trainloader, testloader = utility_functions.load_cifar100(batch_size=200, num_workers=2)
  num_classes = 100

if len(quantized_nets) == 0:
  print(f"ERROR: No model was found in folder {model_foldername}!")
  print(f"Rerun: {syntax}")
  sys.exit(-3)
elif len(quantized_nets) == 1:
  print("INFO: A single model was found. Assuming Full Net")
  result = utility_functions.evaluate(list(quantized_nets.values())[0], trainloader)
  train_loss = result['val_loss']
  train_acc = result['val_acc']
  result = utility_functions.evaluate(list(quantized_nets.values())[0], testloader)
  test_loss = result['val_loss']
  test_acc = result['val_acc']
  print(f"\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")
else:
  print("INFO: Multiple models were found. Assuming Slimmable Net")
  for net in quantized_nets:
    print(f"Evaluating Increment {net}")
    result = utility_functions.evaluate(quantized_nets[net], trainloader)
    train_loss = result['val_loss']
    train_acc = result['val_acc']
    result = utility_functions.evaluate(quantized_nets[net], testloader)
    test_loss = result['val_loss']
    test_acc = result['val_acc']
    print(f"\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")
