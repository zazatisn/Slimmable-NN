import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# load cifar 10 dataset 
# returns two Dataloaders: trainloader and testloader
def load_cifar10(batch_size=64, num_workers=4):
  train_transform = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.RandomRotation(degrees=(0, 10)),
                         #transforms.RandomPerspective(distortion_scale=0.14),
                         #transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
  return trainloader, testloader

# load cifar 100 dataset 
# returns two Dataloaders: trainloader and testloader
def load_cifar100(batch_size=64, num_workers=4):
  train_transform = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.RandomRotation(degrees=(0, 10)),
                         #transforms.RandomPerspective(distortion_scale=0.14),
                         #transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  
  trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

  testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)
  return trainloader, testloader


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels)  
        return loss,acc
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, last_lr: {:.5f}".format(
            epoch+1, result['train_loss'], result['train_accuracy'], result['val_loss'], result['val_acc'], result['lrs'][-1]))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# evaluate slimnet #
# models has each increment in dictionary structure. E.g. "increment1"=model1() #
# Threshold is the minimum acceptable difference of the two most probable classes #
# If threshold is not met then more increments are added #
# test loader is assumed to have batch == 1 #
@torch.no_grad()
def evaluate_slimnet(models, test_loader, threshold):
    for model_name in models.keys():
        models[model_name].eval()
    
    increment_num = len(models)
    accuracy_per_batch = []
    increment_per_batch = []

    for images, labels in test_loader:
        # get 1st increment #
        current_increment_num = 0
        current_increment = f"increment{current_increment_num}"
        model = models[current_increment]
        
        # get the two highest probs and their difference #
        out = model(images)
        out_prob = F.softmax(out, dim=1)
        top2_prob = torch.topk(out_prob, 2, sorted = True).values
        top2_prob_diff = top2_prob[0,0] - top2_prob[0,1] # top2 is 2D because we are dealing with batch #
        acc = accuracy(out, labels)

        while (top2_prob_diff < threshold) and (current_increment_num < increment_num - 1):
            # get next increment #
            current_increment_num += 1
            current_increment = f"increment{current_increment_num}"
            model = models[current_increment]

            # get the two highest probs and their difference #
            out = model(images)
            out_prob = F.softmax(out, dim=1)
            top2_prob = torch.topk(out_prob, 2, sorted = True).values
            top2_prob_diff = top2_prob[0,0] - top2_prob[0,1]
            acc = accuracy(out, labels)
        accuracy_per_batch.append(acc)
        increment_per_batch.append(current_increment_num + 1.0) # start with 1
    test_accuracy = torch.stack(accuracy_per_batch).mean().item()
    average_increment = torch.tensor(increment_per_batch).mean().item()
    return test_accuracy, average_increment


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,weight_decay=0, 
        grad_clip=None, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    for epoch in (range(epochs)):
        # Training Phase 
        model.train()
        train_losses = []
        train_accuracy= []
        lrs=[]
        for (batch_idx, batch) in enumerate(train_loader):
            loss,accuracy = model.training_step(batch)
            train_losses.append(loss)
            train_accuracy.append(accuracy)
            loss.backward()
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            if batch_idx % 60 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.4f}'.
                format(epoch+1, batch_idx , len(train_loader),
                       100. * batch_idx / len(train_loader), loss,accuracy))
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = torch.stack(train_accuracy).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def fit_one_cycle_switchable(epoch, model, train_loader, val_loader, grad_clip=None, opt_func=torch.optim.SGD):
    optimizer = opt_func
    # Training Phase 
    model.train()
    train_losses = []
    train_accuracy= []
    lrs=[]
    for (batch_idx, batch) in enumerate(train_loader):
        loss,accuracy = model.training_step(batch)
        train_losses.append(loss)
        train_accuracy.append(accuracy)
        loss.backward()
        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        lrs.append(get_lr(optimizer))
        # Record & update learning rate
        if batch_idx % 60 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.4f}'.
            format(epoch+1, batch_idx , len(train_loader),
                    100. * batch_idx / len(train_loader), loss,accuracy))
    
    # Validation phase
    result = evaluate(model, val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['train_accuracy'] = torch.stack(train_accuracy).mean().item()
    result['lrs'] = lrs
    model.epoch_end(epoch, result)
    return

def freeze_weights(slimnet_increment):
    for param in slimnet_increment.parameters():
        param.requires_grad = False
    
def unfreeze_weights(slimnet_increment):
    for param in slimnet_increment.parameters():
        param.requires_grad = True

# Note: Check if need to initialize output layers weight accordingly #
# Also, check, if we need to alter the initialize by adding some Gaussian noise with mean 0 and std the weights std #
# Or, initialize the weights based on a gaussian distribution of mean = weights.mean() and std = weights.std() of the previous increment #
def init_weights(slimnet_increments, current_increment, use_noise = True, multiplier = 0.02):
  if current_increment <= 0:
    return
  previous_increment = current_increment - 1
  for name, param in slimnet_increments[current_increment].named_parameters():
    target_name = name.replace(f'subnet{previous_increment}', f'subnet{current_increment}')
    slimnet_increments[current_increment].state_dict()[target_name].copy_(slimnet_increments[current_increment].state_dict()[name])
    if use_noise == True:
      slimnet_increments[current_increment].state_dict()[target_name] += torch.randn(param.shape) * multiplier

def init_weights_channelwise(slimnet_increments, current_increment, use_noise = True, multiplier = 0.02):  
    if current_increment <= 0:
        return
    previous_increment = current_increment - 1
    for name, param in slimnet_increments[current_increment].named_parameters():
        target_name = name
        slimnet_increments[current_increment].state_dict()[target_name].copy_(slimnet_increments[previous_increment].state_dict()[name])
        if use_noise == True:
            slimnet_increments[current_increment].state_dict()[target_name] += torch.randn(param.shape) * multiplier

# copy the weights of the last increment to the first increment #
# applies only for stridewise slimmable networks                #
def copy_weights_stridewsie(slimnet_increments, num_increments):
    for current_increment in range(num_increments - 1):
        for name, _ in slimnet_increments[num_increments - 1].named_parameters():
            slimnet_increments[current_increment].state_dict()[name].copy_(slimnet_increments[num_increments - 1].state_dict()[name])
            

def incremental_training(slimnet_increments, num_increments, trainloader, testloader, epochs, max_lr ,weight_decay=0, grad_clip=None):
    for i in range(num_increments):
        print(f'Training sliment {i}')
        init_weights(slimnet_increments, i, use_noise=False)
        # debug - see which layers are frozen #
        for name, param in slimnet_increments[i].named_parameters():
            print(f"Layer {name}, Requires Grad: {param.requires_grad}")
        opt_func = torch.optim.Adam(slimnet_increments[i].parameters(),max_lr,amsgrad=True, weight_decay=weight_decay)
        fit_one_cycle(epochs, max_lr, slimnet_increments[i], trainloader, testloader, weight_decay, grad_clip, opt_func)
        freeze_weights(slimnet_increments[i])
        
        
def switchable_incremental_training(slimnet_increments, num_increments, trainloader, testloader, epochs, max_lr ,weight_decay=0, grad_clip=None):
    for epoch in range(0, epochs):
        #print(f"Epoch: {epoch}")
        for i in range(num_increments):
            print(f'Training sliment {i}')
            init_weights(slimnet_increments, i, use_noise=False)
            opt_func = torch.optim.Adam(slimnet_increments[i].parameters(),max_lr,amsgrad=True, weight_decay=weight_decay)
            fit_one_cycle_switchable(epoch, slimnet_increments[i], trainloader, testloader, grad_clip, opt_func)
            freeze_weights(slimnet_increments[i])
        for i in range(num_increments):
            unfreeze_weights(slimnet_increments[i])

# perform multiple times #
# stride 4 -> stride 2 -> stride 1 -> stride 4 etc. #
def incremental_training_stridewise(slimnet_increments, num_increments, trainloader, testloader, epochs, max_lr ,weight_decay=0, grad_clip=None, iters=1):
    for _ in range(iters):
        for i in range(num_increments):
            print(f'Training sliment {i}')
            init_weights_channelwise(slimnet_increments, i, use_noise=False)
            opt_func = torch.optim.Adam(slimnet_increments[i].parameters(),max_lr,amsgrad=True, weight_decay=weight_decay)
            fit_one_cycle(epochs, max_lr, slimnet_increments[i], trainloader, testloader, weight_decay, grad_clip, opt_func)
    copy_weights_stridewsie(slimnet_increments, num_increments)

