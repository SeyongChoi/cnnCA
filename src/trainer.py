import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            lattice, ca_int, dL, target = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            output = self.model(lattice, ca_int, dL)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                lattice, ca_int, dL, target = [x.to(self.device) for x in batch]
                output = self.model(lattice, ca_int, dL)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader=None, epochs=100):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
