import os
import sys
import utility_functions
import torch
import alexnet

syntax = "python3 train_alexnet.py (cifar10|cifar100) <batch_size>"

# argument checking #
if len(sys.argv) == 3:
  try:
    batch_size = int(sys.argv[2])
  except:
    print("ERROR: Batch size must be an integer!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  if batch_size <= 0:
    print("ERROR: Batch size must be positive!")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
  dataset = sys.argv[1].lower()
  if dataset != "cifar10" and dataset != "cifar100":
    print(f"ERROR: Invalid dataset {sys.argv[1]}")
    print(f"Rerun: {syntax}")
    sys.exit(-1)
else:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)

if dataset == "cifar10":
  trainloader, testloader = utility_functions.load_cifar10(batch_size=batch_size, num_workers=8)
  num_classes = 10
  epochs = 30
elif dataset == "cifar100":
  trainloader, testloader = utility_functions.load_cifar100(batch_size=batch_size, num_workers=8)
  num_classes = 100
  epochs = 50
device = utility_functions.get_default_device()

print("Training...")
model = utility_functions.to_device(alexnet.AlexNet(num_classes), device)
max_lr = 0.002
grad_clip = None
weight_decay = 0.00004
opt_func = torch.optim.Adam(model.parameters(),max_lr,amsgrad=True
                            , weight_decay=weight_decay)
history = utility_functions.fit_one_cycle(epochs, max_lr, model, trainloader, testloader, 
                                  grad_clip=grad_clip, 
                                  weight_decay=weight_decay, 
                                  opt_func=opt_func)

# save model and check accuracy #
if dataset == "cifar10":
  foldername = f"saved_models/full_alexnet_cifar10/"
  if not os.path.exists(foldername):
      os.makedirs(foldername)
  filename = f"saved_models/full_alexnet_cifar10/model.pt"
  torch.save(model, filename)
elif dataset == "cifar100":
  foldername = f"saved_models/full_alexnet_cifar100/"
  if not os.path.exists(foldername):
      os.makedirs(foldername)
  filename = f"saved_models/full_alexnet_cifar100/model.pt"
  torch.save(model, filename)