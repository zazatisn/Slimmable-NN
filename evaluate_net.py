import os
import sys
import utility_functions
import resnet9_v2
import torch

syntax = "python3 evaluate_net.py <model_foldername> (cifar10|cifar100)"

# store networks in dictionary #
nets = {}

if len(sys.argv) != 3:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)
else:
  model_foldername = sys.argv[1]
  if not os.path.exists(model_foldername):
    print(f"ERROR: Folder {model_foldername} does not exist!")
    print(f"Rerun: {syntax}")
    sys.exit(-2)
  else:
    dir_list = os.listdir(model_foldername)
    for file in dir_list:
      #print(file)
      full_filename = model_foldername + file
      nets[file[:-3]] = torch.load(full_filename, weights_only = False)
  if sys.argv[2].lower() != "cifar10" and sys.argv[2].lower() != "cifar100":
    print(f"ERROR: Invalid dataset {sys.argv[2]}")
    print(f"Rerun: {syntax}")
    sys.exit(-1)

# get dataset in Dataloader #
dataset = sys.argv[2].lower()
if dataset == "cifar10":
  trainloader, testloader = utility_functions.load_cifar10(batch_size=200, num_workers=8)
  num_classes = 10
elif dataset == "cifar100":
  trainloader, testloader = utility_functions.load_cifar100(batch_size=200, num_workers=8)
  num_classes = 100

if len(nets) == 0:
  print(f"ERROR: No model was found in folder {model_foldername}!")
  print(f"Rerun: {syntax}")
  sys.exit(-3)
elif len(nets) == 1:
  print("INFO: A single model was found. Assuming Full Net")
  result = utility_functions.evaluate(list(nets.values())[0], trainloader)
  train_loss = result['val_loss']
  train_acc = result['val_acc']
  result = utility_functions.evaluate(list(nets.values())[0], testloader)
  test_loss = result['val_loss']
  test_acc = result['val_acc']
  print(f"\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")
else:
  print("INFO: Multiple models were found. Assuming Slimmable Net")
  for net in nets:
    print(f"Evaluating Increment {net}")
    result = utility_functions.evaluate(nets[net], trainloader)
    train_loss = result['val_loss']
    train_acc = result['val_acc']
    result = utility_functions.evaluate(nets[net], testloader)
    test_loss = result['val_loss']
    test_acc = result['val_acc']
    print(f"\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")
