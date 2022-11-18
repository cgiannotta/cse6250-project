import time
import torch
from torchmetrics.classification import BinaryF1Score
from tqdm.notebook import tqdm
import numpy as np
import wandb
from pathlib import Path
import os

class run_model():

  def __init__(self, model, optimizer, loss_function, device):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_function
    self.device = device


  def train(self, train_dataloader, val_dataloader=None, epochs = 10):
      """Train the CNN model."""
      
      # Tracking best validation f1-score
      best_f1 = 0

      # Start training loop
      print("Start training...\n")
      print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^9} | {'Train F1':^11} | {'Val F1':^8} | {'Elapsed':^9}")
      print("-"*60)

      results = []

      for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0
        train_acc = []
        train_f1 = []

        # Put the model into the training mode
        self.model.train()
        progress_bar = tqdm(train_dataloader, ascii=True)

        for step, batch in enumerate(progress_bar):
          # Load batch to GPU
          b_embeddings, b_labels = tuple(t.to(self.device) for t in batch)
          b_labels = b_labels.type(torch.LongTensor).to(self.device)

          # Zero out any previously calculated gradients
          self.model.zero_grad()

          # Perform a forward pass. This will return logits.
          logits = self.model(b_embeddings)

          # Compute loss and accumulate the loss values
          loss = self.loss_fn(logits, b_labels)
          total_loss += loss.item()

          # Perform a backward pass to calculate gradients
          loss.backward()

          # Update parameters
          self.optimizer.step()

          # Get the predictions
          preds = torch.argmax(logits, dim=1).flatten()

          # Calculate the accuracy rate
          accuracy = (preds == b_labels).cpu().numpy().mean() * 100
          f1 = BinaryF1Score().to(self.device)
          f1_score = f1(preds, b_labels).to(self.device).cpu().numpy()
          train_acc.append(accuracy)
          train_f1.append(f1_score)

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        train_acc = np.mean(train_acc)
        train_f1 = np.mean(train_f1)
        wandb.log({'avg_train_loss':avg_train_loss})
        wandb.log({'avg_train_f1': train_f1})
        wandb.log({'train_acc':train_acc})

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy, val_f1 = self.evaluate(self.model, val_dataloader, epoch_i)

            # Track the best f1
            if val_f1 > best_f1:
                best_f1 = val_f1

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {train_acc:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {train_f1:^12.6f} | {val_f1:^12.6f} | {time_elapsed:^9.2f}")
            wandb.log({'val_f1': val_f1,'train_f1': train_f1, 'time_elapsed': time_elapsed, 'epoch': epoch_i,'train_loss': avg_train_loss, 'train_accuracy': train_acc,'val_accuracy':val_accuracy,'val_loss':val_loss})

            results.append([epoch_i, avg_train_loss, train_acc, val_loss, val_accuracy])
              
      print("\n")
      print(f"Training complete! Best f1-score: {best_f1:.2f}%.")

      return np.array(results)

  def evaluate(self, model, val_dataloader, epochIdx = 0):
      """After the completion of each training epoch, measure the model's
      performance on our validation set.
      """
      # Put the model into the evaluation mode. The dropout layers are disabled
      # during the test time.
      model.eval()

      # Tracking variables
      val_accuracy = []
      val_f1 = []
      val_loss = []

      # For each batch in our validation set...
      progress_bar = tqdm(val_dataloader, ascii=True)
      for idx, batch in enumerate(progress_bar):
          # Load batch to GPU
          b_embeddings, b_labels = tuple(t.to(self.device) for t in batch)
          b_labels = b_labels.type(torch.LongTensor).to(self.device)

          # init empty np array for labels/preds for epoch
          if (idx == 0):
            labels = np.array([])
            predictions = np.array([])

          # Compute logits
          with torch.no_grad():
              logits = model(b_embeddings)

          # Compute loss
          loss = self.loss_fn(logits, b_labels)
          val_loss.append(loss.item())

          # Get the predictions
          preds = torch.argmax(logits, dim=1).flatten()

          # Calculate the accuracy rate
          accuracy = (preds == b_labels).cpu().numpy().mean() * 100
          labels = np.concatenate((labels, b_labels.cpu().numpy()), axis=0)
          predictions = np.concatenate((predictions, preds.cpu().numpy()), axis=0)
          f1 = BinaryF1Score().to(self.device)
          f1_score = f1(preds, b_labels).to(self.device).cpu().numpy()
          val_accuracy.append(accuracy)
          val_f1.append(f1_score)

      # Compute the average accuracy and loss over the validation set.
      val_loss = np.mean(val_loss)
      val_accuracy = np.mean(val_accuracy)
      val_f1 = np.mean(val_f1)

      return val_loss, val_accuracy, val_f1