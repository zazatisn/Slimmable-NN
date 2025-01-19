import os
import sys
import utility_functions
import torch
import alexnet

syntax = "python3 train_slimmable_alexnet.py <batch_size> <num_increments> <subnetwork_ratio> (cifar10|cifar100)"

if len(sys.argv) != 5:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)
else:
  try:
    batch_size = int(sys.argv[1])
  except:
    print("ERROR: Batch size must be an integer!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  try:
    num_increments = int(sys.argv[2])
  except:
    print("ERROR: Number of Increments must be an integer!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  try:
    subnetwork_ratio = float(sys.argv[3])
  except:
    print("ERROR: Subnetwork Ratio must be a real number!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  if batch_size <= 0 or num_increments <= 0 or subnetwork_ratio <= 0:
    print("ERROR: All arguments must be positive!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  if sys.argv[4].lower() != "cifar10" and sys.argv[4].lower() != "cifar100":
    print(f"ERROR: Invalid dataset {sys.argv[4]}")
    print(f"Rerun: {syntax}")
    sys.exit(-1)

# get dataset in Dataloader #
dataset = sys.argv[4].lower()
print(f"INFO: Training dataset {dataset}")
if dataset == "cifar10":
  trainloader, testloader = utility_functions.load_cifar10(batch_size=batch_size, num_workers=8)
  num_classes = 10
elif dataset == "cifar100":
  trainloader, testloader = utility_functions.load_cifar100(batch_size=batch_size, num_workers=8)
  num_classes = 100

# get dataset in Dataloader #
device = utility_functions.get_default_device() # get device #

# training params
epochs = 30
max_lr = 0.002
grad_clip = None
weight_decay = 0.00004

subnets = []
for i in range(num_increments):
  subnets.append(alexnet.SubAlexNet(num_increments, subnetwork_ratio))

# get each increment for the slimmable network #
slimnet_increments = []
for i in range(num_increments):
  slimnet_increments.append(utility_functions.to_device(alexnet.SlimAlexNetIncrement(num_increments, subnetwork_ratio, i+1, subnets, num_classes), device))

print("Training...")
utility_functions.incremental_training(slimnet_increments, num_increments, trainloader, testloader, epochs, max_lr, weight_decay, grad_clip)

# save each model increment #
if dataset == "cifar10":
  for i in range(num_increments):
    foldername = f"saved_models/slimalexnet_{num_increments}inc_{subnetwork_ratio}ratio_cifar10/"
    if not os.path.exists(foldername):
      os.makedirs(foldername) 
    filename = f"saved_models/slimalexnet_{num_increments}inc_{subnetwork_ratio}ratio_cifar10/increment{i}.pt"
    torch.save(slimnet_increments[i], filename)
elif dataset == "cifar100":
  for i in range(num_increments):
    foldername = f"saved_models/slimalexnet_{num_increments}inc_{subnetwork_ratio}ratio_cifar100/"
    if not os.path.exists(foldername):
      os.makedirs(foldername) 
    filename = f"saved_models/slimalexnet_{num_increments}inc_{subnetwork_ratio}ratio_cifar100/increment{i}.pt"
    torch.save(slimnet_increments[i], filename)