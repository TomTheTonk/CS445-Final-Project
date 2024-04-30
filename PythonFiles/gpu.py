#File that checks if cude if avaible this was used when I was getting pytorch cude running
import torch.cuda

print(torch.version.cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available() )