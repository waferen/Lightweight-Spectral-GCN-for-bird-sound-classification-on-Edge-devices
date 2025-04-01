class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = -float('inf')  # Initialize to negative infinity
        self.epochs_without_improvement = 0

    def __call__(self, val_accuracy):
        if val_accuracy > self.best_score + self.delta:
            self.best_score = val_accuracy  # Update the best score
            self.epochs_without_improvement = 0
            return False  # Continue training
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True  # Stop training
            return False  # Continue training
