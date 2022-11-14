import h5py
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

'''
Retrieve Train and Validation Data Loaders given the H5Py file, CPU/GPU Device, batch size, and the specific phenotype to classify.

'''
def get_data(h5py_file, device, batch_size = 32, phenotype = 0):
  f = h5py.File(h5py_file, 'r')

  train_inputs = torch.from_numpy(f['train'][:])
  train_inputs.to(device)
  
  train_labels = torch.from_numpy(f['train_label'][:,phenotype])
  train_labels.to(device)
  
  train_data = TensorDataset(train_inputs, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  
  val_inputs = torch.from_numpy(f['val'][:])
  val_inputs.to(device)
  
  val_labels = torch.from_numpy(f['val_label'][:,phenotype])
  val_labels.to(device)
  
  val_data = TensorDataset(val_inputs, val_labels)
  val_sampler = SequentialSampler(val_inputs)
  val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
  
  embeddings_tensor = torch.from_numpy(f['w2v'][:])
  embeddings_tensor.to(device)
  
  return train_dataloader, val_dataloader, embeddings_tensor

#Ref: https://discuss.pytorch.org/t/recommend-the-way-to-load-larger-h5-files/32993/9
class Features_Dataset(data.Dataset):
  def __init__(self, archive, phase, phenotype = 0):
    self.archive = archive
    self.phase = phase
    self.phenotype = phenotype
  
  def __getitem__(self, index):
    with h5py.File(self.archive, 'r', libver='latest') as archive:
      embeddings = archive[str(self.phase)][index]
      id = archive[str(self.phase + '_id')][index]
      label = archive[str(self.phase) + '_label'][index, self.phenotype]
      subject = archive[str(self.phase) + '_subj'][index]
      time = archive[str(self.phase) + '_time'][index]
      return embeddings, id, label, subject, time

  def __len__(self):
    with h5py.File(self.archive, 'r', libver='latest') as archive:
      embeddings = archive[str(self.phase)]
      return len(embeddings)