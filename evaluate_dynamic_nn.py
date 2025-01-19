import os
import sys
import utility_functions
import resnet9_v2
import torch

syntax = "python3 evaluate_dynamic_nn.py <model_foldername> <csv_filename> (cifar10|cifar100)"

# store networks in dictionary #
nets = {}

if len(sys.argv) != 4:
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
    csv_filename = f"csv_evaluation_files/{sys.argv[2]}"
  dataset = sys.argv[3].lower()
  if dataset != "cifar10" and dataset != "cifar100":
    print(f"ERROR: Invalid dataset {dataset}")
    print(f"Rerun: {syntax}")
    sys.exit(-2)
    
# get dataset in Dataloader #
if dataset == "cifar10":
  _, testloader = utility_functions.load_cifar10(batch_size=1, num_workers=8)
elif dataset == "cifar100":
  _, testloader = utility_functions.load_cifar100(batch_size=1, num_workers=8)

if len(nets) == 0:
  print(f"ERROR: No model was found in folder {model_foldername}!")
  print(f"Rerun: {syntax}")
  sys.exit(-3)
elif len(nets) == 1:
  print("INFO: A single model was found. Assuming Full Net")
  result = utility_functions.evaluate(list(nets.values())[0], testloader)
  test_loss = result['val_loss']
  test_acc = result['val_acc']
  print(f"\tTest Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")
else:
  print("INFO: Multiple models were found. Assuming Slimmable Net")
  print(f"INFO: Saving Evaluation Metrics in {csv_filename} using csv format!")
  f = open(csv_filename, 'w')
  f.write("Threshold,Test accuracy,Average Increment\n")
  thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99]
  for threshold in thresholds:
    test_accuracy, average_increment = utility_functions.evaluate_slimnet(nets, testloader, threshold)
    print(F"Threshold: {threshold} Test accuracy: {test_accuracy}, Average Increment: {average_increment}")
    f.write(f"{threshold},{test_accuracy},{average_increment}\n")
  f.close()
  

