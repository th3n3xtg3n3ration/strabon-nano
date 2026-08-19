**[← 4. Tokenizasyon](04-tokenization.md) | [İçindekiler](README.md) | Sonraki: [Ek A — Notasyon →](06-notation-and-terms.md)**

---

# 5. Ölçekleme, çözümleme ve doğrulama

## 5.1 Ölçekleme yasaları

### Deneysel biçim

Hoffmann ve ark. (2022), kayıp ile parametre sayısı $N$ ve eğitim token sayısı
$D$ arasındaki ilişkiyi şu biçimde modellemiştir:

$$
\mathcal{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Uydurulan katsayılar: $E = 1{,}69$, $A = 406{,}4$, $B = 410{,}7$,
$\alpha = 0{,}34$, $\beta = 0{,}28$.

Terimlerin yorumu:

- $E$ — indirgenemez terim. Dilin doğal belirsizliğidir; sonsuz model ve sonsuz
  veriyle bile aşağı inilemez.
- $A/N^{\alpha}$ — modelin sınırlı kapasitesinden gelen hata.
- $B/D^{\beta}$ — sınırlı veriden gelen hata.

İki terimin **ayrı** olması kritik sonucu doğurur: yalnızca birini iyileştirmek
kaybı bir tabana dayandırır. Model büyütülüp veri sabit tutulursa
$B/D^{\beta}$ terimi değişmez.

### Hesap kısıtı altında eniyileme

Eğitim maliyeti yaklaşık olarak (§2.11):

$$
C \approx 6ND \quad \text{(FLOP)}
$$

$C$ sabitken $\mathcal{L}$'yi enküçülten $(N, D)$ çifti aranır. Sonuç kuvvet
yasası biçimindedir:

$$
N_{\text{opt}} \propto C^{\,a}, \qquad D_{\text{opt}} \propto C^{\,b}, \qquad a + b = 1
$$

Hoffmann ve ark. bu üsleri üç ayrı yöntemle kestirir ve **yöntemler birbiriyle
tam olarak uyuşmaz.** Bu ayrımın belirtilmesi gerekir:

**Yaklaşım 1 ve 2** (eğitim eğrisi zarfı ve IsoFLOP taraması) $a \approx b
\approx 0{,}5$ verir. Bu durumda $D/N$ hesaptan bağımsız bir sabittir ve
Chinchilla'nın 70 milyar parametre / 1,4 trilyon token noktasından

$$
\frac{D}{N} \approx 20
$$

okunur. Yaygın olarak alıntılanan kural budur.

**Yaklaşım 3** (yukarıdaki parametrik uydurma) ise doğrudan çözüldüğünde

$$
a = \frac{\beta}{\alpha+\beta} = 0{,}457, \qquad b = \frac{\alpha}{\alpha+\beta} = 0{,}544
$$

verir. $a \neq b$ olduğundan bu kestirim altında $D/N$ **sabit değildir**;
hesap büyüdükçe artar:

| $N$ | 20× kuralı | Parametrik uydurmanın verdiği $D$ | İma edilen $D/N$ |
|---|---|---|---|
| 9 M | 180 M | ~107 M | 11,9 |
| 494 M | 9,9 B | ~12,6 B | 25,5 |
| 70 B | 1,4 T | ~4,6 T | 65,6 |

Bu tutarsızlık makalenin kendisine aittir; Besiroglu ve ark. (2024) yeniden
üretme çalışmasında belgelemiştir. Uygulamada 20× kuralı kaba bir yön göstergesi
olarak kullanılır, kesin bir hedef olarak değil.

Kaba yönlendirme amacıyla:

| $N$ | $D \approx 20N$ |
|---|---|
| 9 M | ~180 M token |
| 24 M | ~480 M token |
| 49 M | ~980 M token |
| 494 M | ~9,9 B token |

### Neden pratikte 20'nin üstüne çıkılır

Chinchilla, **eğitim** hesabını eniyiler. Ancak model bir kez eğitilip
defalarca çalıştırılır ve çıkarım maliyeti $D$'ye değil yalnızca $N$'ye
bağlıdır. Toplam yaşam döngüsü maliyeti göz önüne alındığında, daha küçük bir
modeli daha uzun eğitmek çoğu zaman daha iyidir.

Açık model aileleri bu nedenle oranı çok yukarı taşır: Llama 3 8B yaklaşık
1.875 token/parametre, SmolLM2 1,7B yaklaşık 6.470 token/parametre ile
eğitilmiştir.
Strabon'un `--minutes 60` yapılandırmasında ölçülen tipik değer:

```
[budget] 340 ms/step -> 9,847 steps, 645M tokens (71.6 tokens/param)
```

Eğitim betiği bu oranı `--minutes` yolunda yazdırır (sabit adım sayısı
verildiğinde yazdırmaz); 20'nin çok altındaki bir değer modelin eksik
eğitildiğini, çok üstündeki bir değer aynı hesapla daha büyük bir modelin
eğitilebileceğini gösterir.

### Türkçe veri tavanı

Ölçekleme yasası, $D$'nin istenildiği kadar büyütülebildiğini varsayar.
Türkçe için bu varsayım sınırlıdır:

| Korpus | Türkçe hacim |
|---|---|
| FineWeb-2 (`tur_Latn`) | 41,9 milyar sözcük |
| HPLT v2 (`tur_Latn`) | 51,7 milyar sözcük |
| CulturaX (tr) | 64,3 milyar token |

Bu kümeler büyük ölçüde aynı Common Crawl taramalarından türediğinden
birleşimleri toplamlarından belirgin biçimde küçüktür. Kaba bir üst sınır
$10^{11}$ token mertebesindedir; $D/N = 20$ kabulüyle bu, yaklaşık $5$ milyar
parametrelik bir modelin hesap-eniyi eğitimine karşılık gelir. Daha büyük
Türkçe modeller ya yinelenen geçişler (multi-epoch) ya çok dilli karışım ya da
sentetik veri gerektirir.

---

## 5.2 Çözümleme (decoding)

Eğitilmiş model, her konumda $p = \mathrm{softmax}(z)$ dağılımını üretir.
Metin üretmek için bu dağılımdan örnekleme yapılır.

**En olası token'ı seçmek (greedy) neden yetersiz.** $\arg\max$ çözümlemesi
belirlenimcidir ve yüksek olasılıklı döngülere girme eğilimindedir; üretilen
metin tekrarlı olur. Bu, modelin değil çözümleme kuralının kusurudur.

Üç ayar birlikte kullanılır.

### Sıcaklık

$$
p_i(\tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

- $\tau \to 0$: dağılım tek-noktalıya yaklaşır ($\arg\max$).
- $\tau = 1$: modelin ürettiği dağılım değiştirilmeden kullanılır.
- $\tau \to \infty$: düzgün dağılıma yaklaşır.

$\tau \lt 1$ yüksek olasılıkları keskinleştirir, $\tau \gt 1$ düzleştirir.
Varsayılan $0{,}8$.

### Top-$k$ süzmesi

En büyük $k$ logit dışındaki tümü $-\infty$ yapılır ve dağılım yeniden
normalleştirilir. Kuyruk bölgesindeki — tek tek çok düşük olasılıklı ama
toplamda kayda değer kütle taşıyan — token'ların seçilmesini engeller.
Varsayılan $k = 50$.

### Top-$p$ (çekirdek) süzmesi

Olasılığa göre azalan sırada, kümülatif kütlesi $p$'yi aşan **en küçük** küme
tutulur:

$$
S = \arg\min_{S' \subseteq \mathcal{V}} \;\lvert S' \rvert
\quad \text{öyle ki} \quad \sum_{i \in S'} p_i \geq p
$$

(En küçük **eleman sayılı** küme; olasılığa göre azalan sıralamada baştan
alınarak elde edilir.)

Top-$k$'dan farkı, tutulan aday sayısının **dağılıma göre değişmesidir**:
model bir sonraki token'dan eminse birkaç aday, kararsızsa çok sayıda aday
kalır. Varsayılan $p = 0{,}95$.

### Sıralamanın sonucu

Kod üç işlemi şu sırayla uygular: sıcaklık → top-$k$ → top-$p$ → örnekleme.

Sıra önemsiz değildir. Top-$p$, top-$k$'dan **sonra** ve hayatta kalan $k$ aday
üzerinde yeniden normalleştirilmiş dağılımla hesaplanır. Kuyruk kütlesi
atıldığı için kalan adayların olasılıkları yükselir ve kümülatif eşik daha
erken aşılır. Dolayısıyla etkin çekirdek, ham dağılım üzerinde hesaplanacak
çekirdekten **dardır**. Bu, kısıtların birlikte kullanıldığı hemen her
uygulamada geçerli olan standart davranıştır; ayarları seçerken akılda
tutulmalıdır.

---

## 5.3 Doğrulama

Makine öğrenmesi kodundaki hatalar tipik olarak **sessizdir**: program çökmez
ve kayıp yine düşer, ancak model yanlış bir şey öğrenir. Bu nedenle sınamalar
"çalışıyor mu" yerine **sağlanması gereken özellikleri** denetler.

Depodaki 33 sınamanın tamamı çevrimdışı çalışır; veri kümesi veya ağ erişimi
gerektirmez.

```
python -m tests.test_model       10 sınama
python -m tests.test_tokenizer    7 sınama
python -m tests.test_data        16 sınama
```

### Model sınamaları — gerekçeleriyle

| Sınama | Denetlenen özellik | Yakaladığı hata |
|---|---|---|
| İlk kayıp $\approx \log V$ | İlklemenin düzgün dağılım üretmesi | İlkleme ölçeği hatası, hedef sızıntısı |
| Nedensellik | $t$ çıktısının $x_{\gt t}$'den bağımsızlığı | Maske hatası — kayıp *iyileşir*, sınamasız görülmez |
| Tek yığını ezberleme | Gradyan yolunun bütünlüğü | Kopuk gradyan, donmuş parametre |
| Her parametrenin gradyan alması | Ölü bileşen olmaması | Bağlanmamış katman, yanlış `requires_grad` |
| Ağırlık bağlama | $E$ ve çıkış başının aynı bellek | Kopya alınması, tasarrufun kaybı |
| RoPE norm koruması | $\lVert R_m q\rVert = \lVert q \rVert$ | Yanlış açı/eşleştirme, bilgi kaybı |
| GQA yolu | $H_{kv} \lt H$ ile ileri geçiş | Yanlış tekrarlama, boyut hatası |
| Üretim | Geçerli token kimlikleri | Süzme sonrası boş dağılım |
| Bağlam sınırı | $T \gt T_{\max}$ için hata | Sessiz kırpma |
| Parametre formülü | `param_count()` = gerçek sayı | Mimaride belgelenmemiş değişiklik |

**Nedensellik sınaması** özellikle önemlidir. Ölçülen değerler: $t \geq 10$
konumlarındaki token'lar değiştirildiğinde önek çıktılarındaki en büyük fark
$0{,}0$, sonek çıktılarındaki $1{,}27$. İlki sıfır olmalıdır (bağımsızlık),
ikincisi sıfırdan farklı olmalıdır (aksi hâlde sınama hiçbir şey ölçmüyordur).

**Ezberleme sınaması** en hızlı sağlık göstergesidir: sabit bir yığın üzerinde
300 adım sonunda kayıp $6{,}264 \to 0{,}0033$ iner (sınama $V = 512$ kullanır,
dolayısıyla başlangıç $\log 512 = 6{,}24$'tür). İnmiyorsa gradyan zincirinde
bir kopukluk vardır.

### Sayısal türev denetimi

Elle yazılmış geri yayılım kullanılıyorsa, analitik gradyan merkezi sonlu
farkla karşılaştırılmalıdır:

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + h e_i) - \mathcal{L}(\theta - h e_i)}{2h}
$$

Kesme hatası $O(h^2)$, yuvarlama hatası $O(\epsilon_{\text{mak}}/h)$
mertebesindedir; $h$ çok büyükse birincisi, çok küçükse ikincisi baskın olur.
Toplam hatayı enküçülten adım

$$
h^{\*} \approx \epsilon_{\text{mak}}^{1/3}
$$

olup `float64` için ($\epsilon_{\text{mak}} \approx 2{,}2 \times 10^{-16}$)
$h^{\*} \approx 6 \times 10^{-6}$ verir. Bu bölgede bağıl hata $10^{-10}$
mertebesine iner. `float32` kullanılırsa aynı hesap $h^{\*} \approx 5 \times
10^{-3}$ ve bağıl hata $10^{-5}$ verir — bu nedenle türev denetimi her zaman
`float64`'te yapılmalıdır.

Strabon PyTorch'un otomatik türevini kullandığından bu denetim depoda yer
almaz; yöntem, kendi katmanını yazanlar için burada kayda geçirilmiştir.

---

## 5.4 Beklenen başarım ve sınırlar

Bu ölçekte elde edilebilecekler ve edilemeyecekler ayrı ayrı belirtilmelidir.

**Elde edilebilir.** Biçimbirimsel ve sözdizimsel düzenliliklerin öğrenilmesi:
ünlü uyumuna uygun ek seçimi, ünsüz yumuşaması, sözcük sırası, konu
tutarlılığı olan kısa paragraflar.

**Elde edilemez.** Olgusal doğruluk, çok adımlı akıl yürütme, kod üretimi,
soru cevaplama.

Ölçek için karşılaştırma: 561 milyon parametreli ve 11,2 milyar token üzerinde
eğitilmiş bir modelin (Karpathy'nin nanochat *d20* rapor kartı; eğitim-sonrası
ölçüm) MMLU başarımı $0{,}3151$'dir; rastgele tahmin $0{,}25$ verir. Yani
o ölçekte bile olgusal bilgi neredeyse yoktur. `nano-10m` bu modelin
altmışta biri büyüklüğündedir.

Bu bir eksiklik değil, ölçeğin doğal sonucudur. Sıfırdan ön eğitimin bu
projedeki işlevi kullanılabilir bir dil modeli üretmek değil, boru hattının
tamamını uçtan uca kurmak ve ölçmektir. Kullanılabilir bir sistem, açık bir
taban model üzerine eğitim-sonrası (post-training) uygulanarak elde edilir.

---

## 5.5 Kaynakça

- Kaplan ve ark. (2020), *Scaling Laws for Neural Language Models*, arXiv:2001.08361.
- Hoffmann ve ark. (2022), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.
- Holtzman ve ark. (2020), *The Curious Case of Neural Text Degeneration*, arXiv:1904.09751 (top-$p$).
- Fan ve ark. (2018), *Hierarchical Neural Story Generation*, arXiv:1805.04833 (top-$k$).
- Sardana ve ark. (2023), *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448.
- Besiroglu ve ark. (2024), *Chinchilla Scaling: A Replication Attempt*, arXiv:2404.10102.
- Karpathy (2025), *nanochat*, github.com/karpathy/nanochat (d20 rapor kartı).

*Yıl bilgisi arXiv ilk sürüm yılıdır. Kaplan ve ark. (2020) ilk ölçekleme
yasası çalışmasıdır ve §5.1'de kullanılan biçimin öncülüdür.*

---

**[← 4. Tokenizasyon](04-tokenization.md) | [İçindekiler](README.md) | Sonraki: [Ek A — Notasyon →](06-notation-and-terms.md)**

