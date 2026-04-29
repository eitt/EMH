import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from src.models.diffusion.model import ConditionalDiffusionModel, DiffusionProcess
from src.models.evaluation.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        L,
        H,
        target_dim,
        context_dim,
        context_seq_len: int | None = None,
        device='cpu',
        predict_type='noise',
        schedule='linear',
    ):
        self.device = device
        self.L = L
        self.H = H
        self.predict_type = predict_type

        self.model = ConditionalDiffusionModel(
            target_dim,
            context_dim,
            predict_type=predict_type,
            context_seq_len=context_seq_len,
        ).to(device)
        self.diffusion = DiffusionProcess(num_steps=100, schedule=schedule, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            context = X
            
            # Sample t
            t = torch.randint(0, self.diffusion.num_steps, (X.shape[0],), device=self.device).long()
            
            # Add noise to target y
            y_noisy, noise, y_start = self.diffusion.add_noise(y, t)
            
            # Predict
            t_input = t.float().unsqueeze(1)
            pred = self.model(y_noisy, t_input, context)
            
            if self.predict_type == 'noise':
                loss = self.criterion(pred, noise)
            elif self.predict_type == 'x0':
                pred_x0 = self.model.predict_x0(y_noisy, t, pred, self.diffusion.alphas_cumprod)
                loss = self.criterion(pred_x0, y_start)
            else:
                raise ValueError(f"Unknown predict_type: {self.predict_type}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                context = X
                t = torch.randint(0, self.diffusion.num_steps, (X.shape[0],), device=self.device).long()
                y_noisy, noise, y_start = self.diffusion.add_noise(y, t)
                t_input = t.float().unsqueeze(1)
                pred = self.model(y_noisy, t_input, context)
                
                if self.predict_type == 'noise':
                    loss = self.criterion(pred, noise)
                elif self.predict_type == 'x0':
                    pred_x0 = self.model.predict_x0(y_noisy, t, pred, self.diffusion.alphas_cumprod)
                    loss = self.criterion(pred_x0, y_start)
                else:
                    raise ValueError(f"Unknown predict_type: {self.predict_type}")
                    
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs=50):
        os.makedirs('reports/logs', exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'reports/logs/best_diffusion_model.pt')
                logger.debug(f"Epoch {epoch}: New best val loss {val_loss:.6f}")
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")

if __name__ == "__main__":
    L, H = 21, 5
    # For LatAm: 4 tickers * (Returns + Mask + Amihud) = 4 * 3 = 12 channels per lag
    # Context dim = L * 12
    # Target dim = 4
    train_l, val_l = get_dataloaders(L, H)
    
    trainer = Trainer(L, H, target_dim=4, context_dim=L*12)
    trainer.fit(train_l, val_l, epochs=100)
