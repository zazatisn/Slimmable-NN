import torch
import sys
import os
import utility_functions
from neural_compressor.experimental import Quantization
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.metric import METRICS
from neural_compressor.quantization import fit
from neural_compressor.utils.pytorch import load
import shutil

syntax = "python3 quantize_model.py <fp_model_foldername> <quantized_model_foldername> (cifar10|cifar100)"
supported_quantize_methods = "smooth quantization"
nets = {}

if len(sys.argv) != 4:
  print("ERROR: Incorrect number of arguments!")
  print(f"Rerun: {syntax}")
  sys.exit(-1)
else:  
  fp_model_foldername = sys.argv[1]
  if not os.path.exists(fp_model_foldername):
    print(f"ERROR: Folder {fp_model_foldername} does not exist!")
    print(f"Rerun: {syntax}")
    sys.exit(-2)
  
  dir_list = os.listdir(fp_model_foldername)
  for file in dir_list:
    print(f"INFO: Found model {file}")
    
    # construct full path filename #
    if not fp_model_foldername.endswith('/'):
      fp_model_foldername = fp_model_foldername + '/'
    full_filename = fp_model_foldername + file
    
    nets[file[:-3]] = torch.load(full_filename, weights_only = False)
  
  quantized_model_foldername = sys.argv[2]
  if not os.path.exists(quantized_model_foldername):
    os.makedirs(quantized_model_foldername)
  if not quantized_model_foldername.endswith('/'):
    quantized_model_foldername += '/'
    
  if sys.argv[3].lower() != "cifar10" and sys.argv[3].lower() != "cifar100":
      print(f"ERROR: Invalid dataset {sys.argv[3]}")
      print(f"Rerun: {syntax}")
      sys.exit(-1)

# get dataset in Dataloader #
dataset = sys.argv[3].lower()
if dataset == "cifar10":
  trainloader, testloader = utility_functions.load_cifar10(batch_size=1, num_workers=8)
  num_classes = 10
elif dataset == "cifar100":
  trainloader, testloader = utility_functions.load_cifar100(batch_size=1, num_workers=8)
  num_classes = 100

quantizer = Quantization("./quantization_conf.yaml")
quantizer.calib_dataloader = testloader
quantizer.eval_dataloader = testloader

# op_name_dict = {
#   ".*": {
#       "activation": {
#           "dtype": ["int8"],
#           # "algorithm": ["minmax"],
#           # "granularity": ["per_tensor"],
#           # "scheme": ["sym"],
#       },
#       "weight": {
#           "dtype": ["int8"],
#           # "algorithm": ["minmax"],
#           # "granularity": ["per_channel"],
#           # "scheme": ["sym"],
#       },
#   },
# }
# metrics = METRICS("pytorch")
# top1 = metrics["topk"]()
# static_quant_conf = PostTrainingQuantConfig(op_name_dict=op_name_dict)

if len(nets) == 0:
  print(f"ERROR: No model was found in folder {fp_model_foldername}!")
  print(f"Rerun: {syntax}")
  sys.exit(-3)
elif len(nets) == 1: 
  print("[INFO]: A single model was found. Assuming Full Net")
  list(nets.values())[0].eval()
  quantizer.model = list(nets.values())[0]
  #quantized_model = fit(model=list(nets.values())[0], conf=static_quant_conf, calib_dataloader=testloader, eval_dataloader=testloader, eval_metric=top1)
  q_model = quantizer()
  q_model.save(f'{quantized_model_foldername}')
  shutil.move(f'{quantized_model_foldername}best_model.pt', f'{quantized_model_foldername}model.pt')
  int8_model = load(f'{quantized_model_foldername}model.pt', list(nets.values())[0])
  print(int8_model(torch.randn(1, 3, 32, 32)))
  print(list(nets.values())[0](torch.randn(1, 3, 32, 32)))
else:
  for net in nets:
    nets[net].eval()
    quantizer.model = nets[net]
    q_model = quantizer()
    q_model.save(f'{quantized_model_foldername}{net}')
    shutil.move(f'{quantized_model_foldername}{net}/best_model.pt', f'{quantized_model_foldername}{net}.pt')
    shutil.rmtree(f'{quantized_model_foldername}{net}')
    int8_model = load(f'{quantized_model_foldername}{net}.pt', nets[net])
    print(int8_model(torch.randn(1, 3, 32, 32)))
    print(list(nets.values())[0](torch.randn(1, 3, 32, 32)))
    
  
  