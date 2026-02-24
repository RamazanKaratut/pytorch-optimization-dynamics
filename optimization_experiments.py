import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- 1. Veri Hazırlama ---
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# --- 2. Model Tanımı ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

# --- 3. Eğitim ve Test Motoru ---
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler=None, epochs=10, device='cpu'):
    history = {'train_loss': [], 'test_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # O anki Learning Rate'i kaydet
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Test Aşaması
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        avg_val_loss = val_loss / len(test_loader)
        test_acc = 100 * correct / total
        history['test_acc'].append(test_acc)
        
        # Scheduler Adımı (Eğer varsa)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss) # Plateau val_loss'a bakar
            else:
                scheduler.step() # Diğerleri sadece epoch sayar
                
        print(f"  Epoch {epoch+1:02d} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    return history

# --- 4. Görev 1: Optimizer Karşılaştırması ---
def run_optimizer_experiments(device, train_loader, test_loader):
    print("\n" + "="*50)
    print("GÖREV 1: OPTIMIZER KARŞILAŞTIRMASI BAŞLIYOR")
    print("="*50)
    
    epochs = 10
    optimizers_config = {
        'SGD': lambda p: optim.SGD(p, lr=0.01),
        'SGD+Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        'Adam': lambda p: optim.Adam(p, lr=0.001),
        'AdamW': lambda p: optim.AdamW(p, lr=0.001)
    }
    
    results = {}
    for name, opt_fn in optimizers_config.items():
        print(f"\n--- Model eğitiliyor: {name} ---")
        model = SimpleMLP().to(device)
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        results[name] = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=epochs, device=device)
        
    # Görselleştirme ve Kaydetme
    plt.figure(figsize=(15, 5))
    
    # Loss Grafiği
    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history['train_loss'], label=name, marker='o', markersize=4)
    plt.title('Training Loss Karşılaştırması')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 2)
    for name, history in results.items():
        plt.plot(history['test_acc'], label=name, marker='s', markersize=4)
    plt.title('Test Accuracy Karşılaştırması')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    print("\n[BİLGİ] 'optimizer_comparison.png' kaydedildi.")
    return results

# --- 5. Görev 2: LR Scheduler Karşılaştırması ---
def run_scheduler_experiments(device, train_loader, test_loader):
    print("\n" + "="*50)
    print("GÖREV 2: LR SCHEDULER KARŞILAŞTIRMASI BAŞLIYOR")
    print("="*50)
    
    epochs = 15 # Scheduler etkisini görmek için biraz daha uzun tutuyoruz
    base_lr = 0.05 # Scheduler'ların LR'yi nasıl düşürdüğünü görmek için yüksek başlıyoruz
    
    # Scheduler fonksiyonları (Base optimizer olarak SGD+Momentum kullanıyoruz)
    schedulers_config = {
        'Constant (No Scheduler)': lambda opt: None,
        'StepLR': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1),
        'CosineAnnealing': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
        'ReduceLROnPlateau': lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    }
    
    results = {}
    for name, sch_fn in schedulers_config.items():
        print(f"\n--- Model eğitiliyor: {name} ---")
        model = SimpleMLP().to(device)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        scheduler = sch_fn(optimizer)
        criterion = nn.CrossEntropyLoss()
        
        results[name] = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=epochs, device=device)
        
    # Görselleştirme ve Kaydetme
    plt.figure(figsize=(15, 5))
    
    # LR Değişim Grafiği
    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history['lr'], label=name, marker='o', markersize=4)
    plt.title('Learning Rate Değişimi (Epoch Bazlı)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log') # LR farklarını daha iyi görmek için logaritmik ölçek
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 2)
    for name, history in results.items():
        plt.plot(history['test_acc'], label=name, marker='s', markersize=4)
    plt.title('Test Accuracy (Scheduler Etkisi)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scheduler_comparison.png')
    print("\n[BİLGİ] 'scheduler_comparison.png' kaydedildi.")
    return results

# --- 6. Ana Çalıştırma ---
def main():
    torch.manual_seed(42) # Sonuçların tekrarlanabilir olması için
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan Cihaz: {device}")
    
    train_loader, test_loader = get_data()
    
    opt_results = run_optimizer_experiments(device, train_loader, test_loader)
    sch_results = run_scheduler_experiments(device, train_loader, test_loader)
    
    print("\nBütün deneyler başarıyla tamamlandı. Grafikler klasöre kaydedildi!")

if __name__ == "__main__":
    main()  