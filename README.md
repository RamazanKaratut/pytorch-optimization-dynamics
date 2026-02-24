# PyTorch: Optimizasyon Algoritmaları ve Learning Rate Scheduling Analizi

Bu proje, PyTorch kullanılarak aynı Sinir Ağı mimarisinin (MLP) farklı Optimizasyon Algoritmaları (Optimizers) ve Öğrenme Oranı Zamanlayıcıları (LR Schedulers) ile nasıl eğitildiğini karşılaştırmaktadır. 

Amaç, "Hangi optimizer en hızlı yakınsıyor?", "LR scheduler kullanmak performansı nasıl etkiliyor?" gibi teorik sorulara ampirik (deneysel) kanıtlar sunmaktır.

## 📊 Görev 1: Optimizer Karşılaştırması
Bu deneyde aynı model; `SGD`, `SGD+Momentum`, `Adam` ve `AdamW` kullanılarak eğitilmiş ve Loss/Accuracy grafikleri çıkartılmıştır.

![Optimizer Comparison](optimizer_comparison.png)

### Soru & Cevap Analizi:
**1. Hangi optimizer en hızlı yakınsıyor (converge)?**

* Eğrilerden ve eğitim loglarından da net bir şekilde görülebileceği üzere **Adam ve AdamW** en hızlı yakınsayan algoritmalardır. Henüz 2. epoch'ta Adam'ın eğitim kaybı (Train Loss) 0.09 seviyelerine düşerken, ivmesiz klasik SGD 0.29 seviyelerinde kalmıştır. Bunun nedeni, Adam'ın her parametre için öğrenme oranını (learning rate) gradyanların hareketli ortalamasına göre dinamik olarak ayarlamasıdır (Adaptive Learning Rate).

**2. Hangi optimizer en iyi final performansı veriyor?**
* Deney sonuçlarına göre en yüksek final test doğruluklarına **SGD+Momentum (%98.08 peak, %98.02 final)** ve **Adam (%98.00 final)** ulaşmıştır. 
* Klasik SGD ivmesi olmadığı için çok yavaş kalmış (%96.53 final), Adam ise başta çok hızlı öğrenmesine rağmen son epoch'larda SGD+Momentum tarafından yakalanmıştır. Bu da derin öğrenmedeki o meşhur *"Adam çok hızlıdır ama SGD+Momentum daha iyi test performansı verir (geneller)"* kuralının pratik bir ispatıdır.

---

## 📉 Görev 2: Learning Rate Scheduling (Öğrenme Oranı Zamanlama)
Modeli sabit bir Learning Rate ile eğitmek yerine, eğitime yüksek bir LR ile başlayıp minimum noktasına yaklaştıkça LR'yi düşürmek genellikle daha iyi sonuçlar verir. Bu deneyde taban algoritma olarak SGD+Momentum (Başlangıç LR=0.05) kullanılmış ve üç farklı strateji test edilmiştir.

![Scheduler Comparison](scheduler_comparison.png)

### Soru & Cevap Analizi ve Karakteristikler:


**Hangi scheduler en iyi sonuç veriyor? Neden?**
* **CosineAnnealing (%98.57 Test Acc - Kazanan):** Öğrenme oranını bir kosinüs eğrisi şeklinde yavaşça ve pürüzsüzce sıfıra indirdiği için modele en stabil öğrenme sürecini sağlamış ve %98.57 ile en yüksek test doğruluğunu vermiştir.
* **StepLR (%98.41 Test Acc):** Her 5 epoch'ta bir LR'yi %10'una düşürdü. Loglara baktığımızda 5. epoch'ta %97.37 olan başarının, LR düştükten hemen sonra 6. epoch'ta aniden **%98.27**'ye sıçradığı görülmektedir. Ancak ani şoklar nedeniyle CosineAnnealing'in biraz gerisinde kalmıştır.
* **ReduceLROnPlateau (%98.30 Test Acc):** Model 7. epoch'a kadar plato (düzlük) yapmış, ancak 8. epoch'ta scheduler'ın LR'yi düşürmesiyle (0.05 -> 0.025) test başarısı bir anda 97.37'den 98.19'a fırlamıştır.
* **Constant (No Scheduler):** Eğitim boyunca LR sabit (0.05) kaldığı için minimum noktası etrafında sürekli salınım yapmış (overshooting) ve %98.10 ile genel olarak en düşük performansta kalmıştır.

---

## 💻 Kurulum ve Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision pandas numpy matplotlib