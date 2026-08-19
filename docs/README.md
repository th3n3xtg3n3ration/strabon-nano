# Strabon Nano — teknik belgeler

Bu dizi, Strabon Nano'nun kuramsal temelini ve uygulama ayrıntılarını tanımlar.
Her bileşen için üç soru yanıtlanır: **hangi işlem yapılıyor**, **neden bu işlem
seçildi**, **seçimin ölçülebilir sonucu ne**.

Ön koşul: doğrusal cebir, çok değişkenli türev ve temel olasılık. Sinir ağları
hakkında ön bilgi varsayılmaz.

| Bölüm | Kapsam |
|---|---|
| **[1. Problem tanımı ve amaç fonksiyonu](01-problem-ve-amac-fonksiyonu.md)** | Otoregresif ayrıştırma, softmax, çapraz entropi, $\log V$ göstergesi, gradyanın kapalı biçimi, nedensellik kısıtı |
| **[2. Model mimarisi](02-model-mimarisi.md)** | Gömme, RMSNorm, ölçekli nokta-çarpım dikkati ve $\sqrt{d_h}$ türetimi, GQA, RoPE ve göreli konum özelliğinin ispatı, SwiGLU ve $8d/3$ türetimi, artık ilkleme ölçeği, ağırlık bağlama, parametre ve FLOP formülleri |
| **[3. Eğitim yordamı](03-egitim-yordami.md)** | AdamW güncelleme denklemleri ve sapma düzeltmesi, ayrıştırılmış sönüm, ısınma + kosinüs çizelgesi, gradyan kırpma, biriktirme, karma hassasiyet ve kayıp ölçekleme, duvar saati bütçesi |
| **[4. Tokenizasyon ve Türkçe'nin getirdiği kısıtlar](04-tokenizasyon.md)** | BPE algoritması, bayt düzeyi kodlamanın sonuçları, ön-parçalama, doğurganlık ve %TR ölçütleri, Türkçe'nin eklemeli yapısı, veri süzgeçleri |
| **[5. Ölçekleme, çözümleme ve doğrulama](05-olcekleme-ve-dogrulama.md)** | Chinchilla yasası ve türetimi, Türkçe veri tavanı, sıcaklık / top-$k$ / top-$p$, 33 sınamanın gerekçeleri, beklenen başarım sınırları |
| **[Ek A. Notasyon ve terimler](A-notasyon.md)** | Sembol tablosu, terim sözlüğü, kavram–kod eşlemesi |

---

## Okuma yolları

**Mimariyi anlamak için:** 1 → 2. Bu iki bölüm modelin ne hesapladığını tam
olarak tanımlar.

**Eğitimi çalıştırmak veya ayarlamak için:** 3 → 5.1. Eniyileyici, çizelge ve
ölçekleme ilişkisi burada.

**Türkçe'ye özgü kararlar için:** 4. Tokenizasyon ve veri hattı.

**Kodu değiştirmeden önce:** 5.3. Sınamaların neyi güvence altına aldığı.

---

## Gösterim

Denklemler LaTeX ile yazılmıştır ve GitHub ile çoğu Markdown görüntüleyicide
biçimlenir. Biçimlenmeyen ortamlarda kaynak metin okunabilir kalacak biçimde
sade tutulmuştur.

Sayısal değerler, aksi belirtilmedikçe bu depodaki koddan ölçülmüştür.
Dış kaynaklı değerler ilgili bölümün kaynakçasında künyelenmiştir.
