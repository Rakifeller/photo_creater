# AI Selfie Backend

Bu proje, kullanıcıların kendi selfielerini yükleyip AI ile kendilerine benzeyen görseller üretmesini sağlar.  
**FastAPI + Diffusers (SDXL + IP-Adapter Face)** ile geliştirilmiştir ve GPU üzerinde çalışacak şekilde tasarlanmıştır.  

## 🚀 Özellikler
- **SDXL + IP-Adapter Plus Face**: Yüksek yüz benzerliği ile görsel üretim
- **Precompute Embeddings**: Kimlik bilgilerini önceden çıkararak tekrar üretimlerde hız
- **REST API**: Kolay entegrasyon için JSON tabanlı API
- **Runpod & Docker uyumlu**: Cloud GPU üzerinde çalıştırılabilir
- **LCM LoRA opsiyonu**: Hızlı üretim için
