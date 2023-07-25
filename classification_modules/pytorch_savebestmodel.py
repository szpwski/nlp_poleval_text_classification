"""
Module containing class to save model with best epoch
"""

import torch

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), best_deviation=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.best_deviation = best_deviation
        
    def __call__(
        self, current_valid_loss, current_train_loss,
        epoch, model, optimizer, criterion, train_scores, val_scores,
        custom_model_name
    ):
        current_deviation = torch.absolute(current_valid_loss - current_train_loss)
        if (current_valid_loss < self.best_valid_loss) & (current_deviation <= self.best_deviation):
            self.best_valid_loss = current_valid_loss
            self.best_deviation = current_deviation
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nBest deviation: {self.best_deviation}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            for k in train_scores.keys():
                print(f"\nTrain {k}:{train_scores.get(k): .4f} | Validation {k}:{val_scores.get(k): .4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'models/' + custom_model_name + '_epoch_' + str(epoch) + '.pth')