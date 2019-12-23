import torch
import torch.nn as nn 

def CE_loss(logit, label):
    '''
    简单交叉熵损失
    torch版本
    '''
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = criterion(logit, label.long())
    return total_loss 