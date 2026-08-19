---

**[← 5. Ölçekleme ve doğrulama](05-olcekleme-ve-dogrulama.md) | [İçindekiler](README.md)**

---

# Ek A — Notasyon ve terimler

## A.1 Semboller

| Sembol | Anlam | Bölüm |
|---|---|---|
| $V$ | sözlük büyüklüğü | 1.1 |
| $T$ | dizi uzunluğu / bağlam penceresi | 1.1 |
| $S$ | toplam eğitim adımı sayısı | 3.2 |
| $S_w$ | ısınma adım sayısı | 3.2 |
| $B$ | mikro-yığın büyüklüğü (dizi sayısı) | 3.4 |
| $A$ | gradyan biriktirme adımı | 3.4 |
| $d$ | model boyutu | 2.1 |
| $L$ | katman sayısı | 2.10 |
| $H$ | dikkat başlığı sayısı | 2.3 |
| $H_{kv}$ | anahtar-değer başlığı sayısı (GQA) | 2.4 |
| $d_h = d/H$ | başlık boyutu | 2.3 |
| $f$ | ileri besleme ara boyutu | 2.6 |
| $N$ | parametre sayısı | 2.10 |
| $D$ | eğitim token sayısı | 5.1 |
| $C$ | eğitim hesabı (FLOP) | 5.1 |
| $\mathcal{L}$ | kayıp (negatif log-olabilirlik) | 1.3 |
| $z$ | logit vektörü | 1.2 |
| $p$ | softmax çıktısı, olasılık dağılımı | 1.2 |
| $\theta$ | tüm model parametreleri | 1.3 |
| $g_t$ | $t$. adımdaki gradyan | 3.1 |
| $\eta_t$ | $t$. adımdaki öğrenme oranı | 3.2 |
| $\eta_{\max}, \eta_{\min}$ | çizelgenin uç değerleri | 3.2 |
| $\varepsilon$ | sayısal kararlılık sabiti (RMSNorm, AdamW) | 2.2, 3.1 |
| $\Delta t$ | ölçülen adım süresi | 3.7 |
| $\lambda$ | ağırlık sönümü katsayısı | 3.1 |
| $\beta_1, \beta_2$ | AdamW moment katsayıları | 3.1 |
| $\rho$ | $\eta_{\min}/\eta_{\max}$ oranı | 3.2 |
| $c$ | gradyan kırpma eşiği | 3.3 |
| $S$ | kayıp ölçekleme çarpanı (fp16) | 3.6 |
| $E$ | gömme matrisi $\mathbb{R}^{V\times d}$ | 2.1 |
| $R_m$ | $m$ konumu için RoPE döndürme matrisi | 2.5 |
| $\theta_i$ | RoPE taban frekansı | 2.5 |
| $b$ | RoPE taban sabiti ($10^4$) | 2.5 |
| $h^{(\ell)}$ | $\ell$. katmanın çıktısı (artık akış) | 2 |
| $\alpha, \beta$ | ölçekleme yasası üsleri | 5.1 |
| $M$ | nedensel maske | 2.3 |
| $F$ | doğurganlık (token/sözcük) | 4.3 |
| $\tau$ | çözümleme sıcaklığı | 5.2 |

**Sembol çakışmaları.** Alan yazınının yerleşik gösterimini korumak için bazı
harfler birden çok anlamda kullanılmıştır. Karışma riski taşıyanlar:

| Harf | Anlamlar | Ayrım |
|---|---|---|
| $A$ | dikkat ağırlık matrisi (§2.3); biriktirme adımı (§3.4); ölçekleme katsayısı (§5.1) | bağlamdan |
| $B$ | mikro-yığın (§3.4); ölçekleme katsayısı (§5.1) | bağlamdan |
| $E$ | gömme matrisi (§2.1); indirgenemez kayıp (§5.1) | bağlamdan |
| $F$ | alt katman fonksiyonu (§2.7); doğurganlık (§4.3) | bağlamdan |
| $b$ | RoPE tabanı (§2.5); ölçekleme üssü (§5.1) | bağlamdan |
| $\theta$ | model parametreleri; $\theta_i$ RoPE frekansı | alt indis |

$T$ (bağlam) ile $S$ (toplam adım) bilinçli olarak ayrı harflerle
gösterilmiştir; alan yazınında ikisi de $T$ ile anılır ve bu, adım başına
token hesabında karışıklık yaratır.

---

## A.2 Terimler

**Ağırlık bağlama** *(weight tying)* — Çıkış izdüşümünün gömme matrisinin
devriği olarak alınması, $z = h E^{\top}$. Küçük modellerde parametrenin büyük
kısmı gömmede olduğundan kayda değer tasarruf sağlar. → 2.8

**Ağırlık sönümü** *(weight decay)* — Parametreleri her adımda sıfıra doğru
çeken düzenlileştirme terimi. AdamW'de gradyandan ayrıştırılmış olarak
uygulanır. → 3.1

**Aşırı uyum** *(overfitting)* — Modelin genelleme yerine eğitim verisini
ezberlemesi. Belirtisi, eğitim kaybı düşerken doğrulama kaybının yükselmesidir.
Tek geçişli ön eğitimde nadirdir. → 3.8

**Bayt çifti kodlaması** *(BPE)* — En sık bitişik simge çiftini yinelemeli
olarak birleştirerek alt-sözcük sözlüğü öğrenen algoritma. → 4.2

**Çapraz entropi** — İki dağılım arasındaki uyumsuzluk ölçüsü. Hedef
tek-noktalı olduğunda negatif log-olabilirliğe indirgenir. → 1.3

**Çözümleme** *(decoding)* — Eğitilmiş modelin ürettiği dağılımdan token
dizisi örnekleme süreci. → 5.2

**Dikkat** *(attention)* — Her konumun geçmiş konumlardan ağırlıklı bilgi
topladığı işlem, $\operatorname{softmax}(QK^{\top}/\sqrt{d_h} + M)\,V\,W_O$. → 2.3

**Doğrulama kümesi** *(validation set)* — Eğitimde kullanılmayan, genelleme
başarımını ölçmeye ayrılmış veri parçası. → 3.8

**Doğurganlık** *(fertility)* — Sözcük başına düşen ortalama token sayısı.
Tokenizer sıkıştırma verimini ölçer. → 4.3

**Eklemeli dil** *(agglutinative language)* — Dilbilgisel işlevlerin köke
sırayla eklenen, her biri tek işlev taşıyan biçimbirimlerle kurulduğu dil
türü. Türkçe bu sınıftadır. → 4.4

**Epok** *(epoch)* — Eğitim kümesinin bir kez baştan sona geçilmesi. Bu
projede korpus bütçeden büyük olduğu için epok kavramı kullanılmaz; örnekleme
yerine koymalıdır. → 3.5

**Eniyileyici** *(optimizer)* — Gradyanlardan parametre güncellemesi üreten
algoritma. Bu projede AdamW. → 3.1

**FlashAttention** — Dikkat matrisini belleğe hiç yazmadan, blok blok
hesaplayan çekirdek. Bellek kullanımını $O(T^2)$'den $O(T)$'ye indirir.
PyTorch uygun donanımda otomatik olarak buna düşer. → 2.3

**Gömme** *(embedding)* — Ayrık token kimliklerini yoğun vektörlere eşleyen
öğrenilmiş tablo. → 2.1

**Gradyan** — Kaybın parametrelere göre kısmi türevleri vektörü; kaybı en hızlı
artıran yönü verir. → 1.4

**Gradyan biriktirme** — Etkin yığını mikro-yığınlara bölerek gradyanları
toplama tekniği. Bellek gereksinimini böler, sonucu değiştirmez. → 3.4

**Gradyan kırpma** — Gradyan normunu bir eşiğe indirerek tek bir adımın modeli
bozmasını engelleme. → 3.3

**Isınma** *(warmup)* — Öğrenme oranının eğitim başında sıfırdan doğrusal
olarak yükseltilmesi. → 3.2

**Karma hassasiyet** *(mixed precision)* — Eğitimin bir bölümünün düşük
hassasiyetli biçimlerde (bf16/fp16) yürütülmesi. → 3.6

**Kayıp** *(loss)* — Enküçültülen amaç fonksiyonu; burada negatif
log-olabilirlik. Eğitilmemiş modelde $\log V$ olmalıdır. → 1.3

**KV önbelleği** — Çözümleme sırasında geçmiş konumların anahtar ve değer
vektörlerinin saklanması. Bellek maliyeti $T$ ile doğrusal büyür. → 2.4

**Nedensellik** *(causality)* — Modelin $t$ konumunda $x_{\geq t}$'yi
görmemesi kısıtı. Dikkat maskesiyle uygulanır. → 1.5, 2.3

**Ölçekleme yasası** *(scaling law)* — Kaybın parametre ve veri sayısına
bağımlılığını veren ampirik ilişki. → 5.1

**Otoregresif** — Her öğenin yalnızca kendinden öncekilere koşullu olarak
modellenmesi. → 1.2

**Ön-norm** *(pre-norm)* — Normalizasyonun alt katmandan önce uygulanması.
Artık yolunu normalizasyondan bağımsız bırakır. → 2

**Ön-parçalama** *(pre-tokenization)* — BPE'den önce metnin kaba birimlere
bölünmesi; birleştirmelerin sözcük sınırlarını aşmasını engeller. → 4.2

**Parametre** — Eğitim sırasında ayarlanan model değişkenlerinden biri. → 2.10

**RMSNorm** — Karekök-ortalama-kare normalizasyonu; LayerNorm'un ortalama
çıkarmayan, kaydırmasız değişkesi. → 2.2

**RoPE** — Sorgu ve anahtar vektörlerini konuma bağlı açıyla döndüren konum
kodlaması. Dikkat skorunu göreli uzaklığın fonksiyonu yapar. → 2.5

**Sıcaklık** *(temperature)* — Çözümlemede logit'lerin bölündüğü katsayı;
dağılımın keskinliğini denetler. → 5.2

**Softmax** — Gerçel skorları olasılık dağılımına dönüştüren fonksiyon. → 1.2

**SwiGLU** — Kapılı ileri besleme katmanı,
$(\mathrm{SiLU}(xW_1) \odot xW_3)W_2$. → 2.6

**Şaşkınlık** *(perplexity)* — $\exp(\mathcal{L})$. Modelin her konumda kaç eşit
olasılıklı seçenek arasında kararsız olduğunun ölçüsü. → 1.3

**TF32** — Ampere ve sonrası donanımda bulunan matris çarpımı biçimi. fp32'nin
üs aralığını korur, mantisi 10 bite indirir. Kod bunu etkinleştirir; biriktirme
fp32'de kaldığı için kararlılık etkilenmez. → 3.6

**Token** — Modelin üzerinde çalıştığı ayrık birim; alt-sözcük düzeyindedir. → 4.1

**Top-$k$ / top-$p$** — Çözümlemede aday kümesini sınırlayan iki süzme yöntemi;
biri sabit sayıda, diğeri kümülatif olasılığa göre. → 5.2

**Transformer** *(dönüştürücü)* — Dikkat mekanizması üzerine kurulu mimari
(Vaswani ve ark., 2017). → 2

**Yığın** *(batch)* — Bir gradyan kestirimi için birlikte işlenen dizi kümesi.
Koddaki `batch_size` mikro-yığındır; etkin yığın biriktirmeyle çarpılır. → 3.4

---

## A.3 Uygulama karşılıkları

| Kavram | Dosya | Sınıf / fonksiyon |
|---|---|---|
| Yapılandırma, parametre sayımı | `strabon/config.py` | `ModelConfig`, `TrainConfig` |
| Veri süzme, tokenize, yükleme | `strabon/data.py` | `keep`, `tokenize`, `BinaryLoader` |
| BPE eğitimi, ölçütler | `strabon/tokenizer.py` | `train_tokenizer`, `measure` |
| Normalizasyon | `strabon/model.py` | `RMSNorm` |
| Konum kodlaması | `strabon/model.py` | `rope_tables`, `apply_rope` |
| Dikkat | `strabon/model.py` | `Attention` |
| İleri besleme | `strabon/model.py` | `SwiGLU` |
| Tam model | `strabon/model.py` | `Strabon` |
| Öğrenme oranı çizelgesi | `strabon/train.py` | `learning_rate` |
| Eğitim döngüsü | `strabon/train.py` | `train` |
| Süre bütçesi | `strabon/train.py` | `_benchmark_step_time` |
| Çözümleme | `strabon/model.py` | `Strabon.generate` |
| Kontrol noktası yükleme, örnekleme arayüzü | `strabon/sample.py` | `load_checkpoint`, `main` |

---

**[← 5. Ölçekleme ve doğrulama](05-olcekleme-ve-dogrulama.md) | [İçindekiler](README.md)**
