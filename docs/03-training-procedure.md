**[← 2. Model mimarisi](02-model-architecture.md) | [İçindekiler](README.md) | Sonraki: [4. Tokenizasyon →](04-tokenization.md)**

---

# 3. Eğitim yordamı

Eğitim, §1.3'teki $\mathcal{L}(\theta)$'yi enküçültmektir. Bu bölüm kullanılan
eniyileyiciyi, çizelgeyi ve sayısal uygulamayı tanımlar.

---

## 3.1 Eniyileyici: AdamW

Düz gradyan inişi ($\theta \leftarrow \theta - \eta g$) dil modeli eğitiminde
kullanılmaz: parametrelerin gradyan büyüklükleri mertebeler arası farklılık
gösterir ve tek bir $\eta$ hepsine uymaz.

AdamW her parametre için birinci ve ikinci moment kestirimleri tutar:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\[2pt]
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{2} \\[2pt]
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\[4pt]
\theta_t &= \theta_{t-1} - \eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda\,\theta_{t-1}\right)
\end{aligned}
$$

Varsayılan hiperparametreler (`config.py: TrainConfig`):

| Sembol | Kod alanı | Değer |
|---|---|---|
| $\eta_{\max}$ | `lr` | $6 \times 10^{-4}$ |
| $\rho = \eta_{\min}/\eta_{\max}$ | `min_lr_ratio` | 0,1 |
| $\beta_1$ | `beta1` | 0,9 |
| $\beta_2$ | `beta2` | 0,95 |
| $\lambda$ | `weight_decay` | 0,1 |
| $\varepsilon$ | (PyTorch varsayılanı) | $10^{-8}$ |
| $c$ | `grad_clip` | 1,0 |
| $B$ (mikro-yığın) | `batch_size` | 16 |
| $A$ (biriktirme) | `grad_accum` | 8 |
| doğrulama sıklığı | `eval_every` / `eval_steps` | 500 / 50 |
| kontrol noktası sıklığı | `save_every` | 2.000 |
| tohum | `seed` | 1337 |

**Her terimin işlevi.**

- $m_t$ — gradyanın üstel ağırlıklı ortalaması. Yığından yığına gelen gürültüyü
  düzler; etkisi bir eylemsizlik (momentum) terimidir.
- $v_t$ — gradyan karesinin ortalaması. Bölme işlemi güncellemeyi **ölçek
  değişmez** kılar: gradyanı sürekli büyük olan parametre küçük adım, küçük
  olan büyük adım alır. Etkin adım büyüklüğü kabaca $\eta$ mertebesindedir.
- Sapma düzeltmesi — $m_0 = v_0 = 0$ olduğundan ilk adımlarda kestirimler
  sıfıra doğru yanlıdır. $\mathbb{E}[m_t] = (1-\beta_1^{t})\,\mathbb{E}[g]$
  olduğu için $(1-\beta_1^{t})$'ye bölmek bu yanlılığı giderir. Düzeltme
  olmadan ilk adımlar gereğinden küçük kalır.
- $\beta_2 = 0{,}95$ — varsayılan 0,999 yerine. Dil modeli eğitiminde gradyan
  istatistikleri hızlı değişir; kısa pencere daha duyarlı davranır. GPT-2'den
  bu yana yaygın seçimdir.

### Neden "W": ayrıştırılmış ağırlık sönümü

Klasik L2 düzenlileştirmede ceza terimi gradyana eklenir: $g \leftarrow g + \lambda\theta$.
Bu, Adam'da istenmeyen bir sonuç doğurur — ceza da $\sqrt{\hat{v}}$'ye bölünür,
dolayısıyla gradyan geçmişi büyük olan parametreler **daha az** sönüm alır.
Düzenlileştirmenin amacı bu değildir.

AdamW sönümü güncelleme kuralına ayrı bir terim olarak yazar (yukarıdaki
$\lambda\theta_{t-1}$), böylece sönüm oranı gradyan istatistiklerinden bağımsız
kalır (Loshchilov ve Hutter, 2019).

### Hangi parametrelere sönüm uygulanır

Yalnızca en az iki boyutlu tensörlere ($W_Q, W_K, W_V, W_O, W_1, W_2, W_3, E$).
RMSNorm kazançları tek boyutludur ve sönüm dışı bırakılır.

**Gerekçe.** Sönüm, kapasiteyi sınırlamak için ağırlık matrislerini küçültür.
Normalizasyon kazançları kapasite taşımaz; onları sıfıra doğru itmek yalnızca
normalizasyonun ölçeğini bozar. Kod bu ayrımı `configure_optimizer` içinde
`p.dim() >= 2` ölçütüyle yapar.

---

## 3.2 Öğrenme oranı çizelgesi

$\eta$ sabit tutulmaz. Kullanılan çizelge doğrusal ısınma ve kosinüs sönümün
birleşimidir. $S_w$ ısınma adımı, $S$ toplam adım sayısı olmak üzere
(bağlam uzunluğu $T$ ile karışmaması için ayrı sembol kullanılmıştır):

$$
\eta_t =
\begin{cases}
\eta_{\max} \cdot \dfrac{t+1}{S_w}, & t < S_w \\[10pt]
\eta_{\min} + \dfrac{1}{2}\left(1 + \cos\dfrac{\pi (t - S_w)}{S - S_w}\right)\left(\eta_{\max} - \eta_{\min}\right), & S_w \leq t < S \\[10pt]
\eta_{\min}, & t \geq S
\end{cases}
$$

burada $\eta_{\min} = \rho\,\eta_{\max}$. Üçüncü durum, `--resume` ile bütçenin
aşıldığı hâlleri kapsar; olmasaydı kosinüs $\pi$'yi geçer ve $\eta$ yeniden
yükselirdi.

**Isınma neden gerekli.** $t=0$'da parametreler rastgeledir ve gradyanlar
büyüktür; ayrıca AdamW'nin $v_t$ kestirimi henüz güvenilmezdir. Tam $\eta$ ile
atılan ilk adımlar modeli kararsız bir bölgeye taşır ve kayıp ıraksar.
Kullanılan oran $S_w = 0{,}03\,S$'dir (`--minutes` yolunda; sabit adım sayısı
verildiğinde varsayılan 500/20.000 = %2,5).

**Sönüm neden gerekli.** Eğitim ilerledikçe parametreler iyi bir bölgeye
yaklaşır; sabit büyüklükte adımlar bu bölgenin etrafında salınım üretir ve
yakınsamayı engeller. Kosinüs biçimi, doğrusal sönüme göre sonda daha uzun süre
küçük $\eta$'da kalır ve ampirik olarak biraz daha düşük son kayıp verir.

**$\eta_{\min} > 0$ neden.** Tam sıfıra inmek öğrenmeyi sonda tamamen durdurur
ve çizelgenin son bölümünü boşa harcar. $\rho = 0{,}1$ yaygın bir seçimdir.

---

## 3.3 Gradyan kırpma

Bir yığın olağandışı büyük gradyan üretirse tek bir adım modeli bozabilir.
Kırpma, gradyan vektörünün küresel normunu bir üst sınıra çeker:

$$
\hat{g} = g \cdot \min\left(1, \frac{c}{\lVert g \rVert_2}\right), \qquad c = 1{,}0
$$

Norm **tüm parametreler üzerinde birlikte** hesaplanır (parametre başına değil),
böylece gradyanın **yönü** korunur, yalnızca büyüklüğü sınırlanır.

Kırpma, karma hassasiyette ölçek geri alındıktan sonra uygulanmalıdır (§3.6);
aksi hâlde eşik ölçek çarpanına göre anlamsız hâle gelir. Kod bu sırayı
`scaler.unscale_(optimizer)` çağrısıyla sağlar.

---

## 3.4 Yığınlama ve gradyan biriktirme

Tek bir diziden hesaplanan gradyan, gerçek gradyanın yüksek varyanslı bir
kestirimidir. $B$ dizinin ortalaması varyansı $1/B$ oranında düşürür.

Bellek, $B$'yi doğrudan büyütmeyi engeller. Gradyan biriktirme, $B_{\text{etkin}}$
diziyi $A$ mikro-yığına böler ve gradyanları toplar:

$$
\nabla \mathcal{L} = \frac{1}{A}\sum_{a=1}^{A} \nabla \mathcal{L}_a
$$

Kod her mikro-yığının kaybını `loss / grad_accum` ile ölçekleyip geri yayar;
PyTorch gradyanları varsayılan olarak biriktirdiği için sonuç, tek seferde
$B_{\text{etkin}}$ dizi işlemekle **matematiksel olarak özdeştir**.

Varsayılan yapılandırma:

| Büyüklük | Değer |
|---|---|
| Mikro-yığın (`batch_size`) | 16 dizi |
| Biriktirme (`grad_accum`) | 8 |
| **Etkin yığın** | **128 dizi** |
| Bağlam ($T$) | 512 |
| **Adım başına token** | **65.536** |

Bellekte hiçbir zaman 16 diziden fazlası bulunmaz.

---

## 3.5 Veri yükleme

Tokenize edilmiş korpus, tek boyutlu bir `uint16` dizisi olarak diske yazılır
($V < 65536$ için; büyük sözlüklerde `uint32`). Yükleyici bu dosyayı `np.memmap`
ile açar: dosya belleğe kopyalanmaz, işletim sistemi yalnızca dokunulan
sayfaları getirir. Bu sayede korpus RAM'den büyük olabilir.

Her adımda $B_{\text{etkin}}$ değil, mikro-yığın büyüklüğü kadar rastgele başlangıç indisi seçilir ve

$$
x = s[i : i{+}T], \qquad y = s[i{+}1 : i{+}1{+}T]
$$

pencereleri alınır. $y$, $x$'in bir konum kaydırılmış hâlidir — §1.2'deki
"bir sonraki token" hedefi budur. `tests/test_data.py` bu kaymayı doğrular.

**İki özellik açıkça belirtilmelidir:**

1. Örnekleme **yerine koymalıdır**; bir epok kavramı yoktur. Korpus eğitim
   bütçesinden büyük olduğunda çoğu token bir kez bile görülmez.
2. Pencereler **belge sınırlarını aşabilir**. Her belgenin sonuna `<|eos|>`
   token'ı yazılır; model bu token'dan sonra bağlamı sıfırlamayı öğrenir. Bu,
   nanoGPT'den bu yana standart yaklaşımdır.

---

## 3.6 Karma hassasiyet

Kayan noktalı biçimlerin bit dağılımı:

| Biçim | İşaret | Üs | Mantis | Yaklaşık aralık |
|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | $10^{-38} \dots 10^{38}$ |
| fp16 | 1 | 5 | 10 | $6{\times}10^{-5} \dots 6{,}5{\times}10^{4}$ |
| bf16 | 1 | 8 | 7 | fp32 ile aynı |

**bf16**, mantis bitlerini üsse feda eder: hassasiyeti fp16'dan düşük ama
aralığı fp32 ile aynıdır. Eğitim için bu doğru ödünleşimdir, çünkü sorun
hassasiyet değil aralıktır.

**fp16 ve alt taşma.** Dil modeli eğitiminde gradyanların büyük kısmı
$10^{-7}$ mertebesindedir ve fp16'nın alt sınırının altına düşerek **sıfıra
yuvarlanır**. Bu, o parametrelerin hiç güncellenmemesi demektir.

Çözüm, kaybı bir $S$ çarpanıyla ölçeklemektir. Zincir kuralı doğrusal olduğu
için tüm gradyanlar $S$ katına çıkar ve temsil edilebilir aralığa taşınır:

$$
\tilde{g} = \nabla(S \cdot \mathcal{L}) = S \cdot \nabla \mathcal{L}
$$

Eniyileyici adımından önce ölçek geri alınır. `torch.amp.GradScaler` $S$'yi
dinamik ayarlar: taşma (`inf`/`NaN`) görülürse adımı atlar ve $S$'yi yarıya
indirir; belirli sayıda temiz adımdan sonra artırır.

**bf16'da ölçekleyiciye gerek yoktur** — aralık zaten yeterlidir. Kod donanımı
sorgular ve otomatik seçer:

| Donanım | Seçilen |
|---|---|
| Ampere ve sonrası (A100, RTX 30/40, L4) | bf16, ölçekleyici kapalı |
| Turing ve öncesi (T4, P100, RTX 20) | fp16 + GradScaler |
| GPU yok | fp32 |

Kaggle'ın T4'ü Turing mimarisidir ve bf16 desteklemez; ikinci satır uygulanır.

**TF32 notu.** Kod ayrıca `torch.backends.cuda.matmul.allow_tf32 = True`
ayarlar. TF32, Ampere ve sonrasında bulunan bir matris çarpımı biçimidir:
fp32'nin üs aralığını korur ama mantisi 10 bite indirir. Dolayısıyla tablodaki
"fp32" satırı, uygun donanımda çarpımlar için fiilen TF32 demektir. Biriktirme
fp32'de yapıldığından eğitim kararlılığı etkilenmez; kazanç hızdır.

---

## 3.7 Duvar saati bütçesi

Kosinüs çizelgesi $S$'yi (toplam adım) önceden bilmeyi gerektirir. Kullanıcı
adım sayısı yerine süre verdiğinde (`--minutes`), $S$ ölçümle belirlenir:

1. 12 adım ısınma amaçlı çalıştırılır ve **atılır** (ilk adımlar bellek
   ayırma ve çekirdek seçimi nedeniyle yavaştır).
2. 15 adım ölçülür; adım süresi $\Delta t$ elde edilir ($\tau$ sembolü
   çözümleme sıcaklığı için ayrılmıştır).
3. $S = \left\lfloor \dfrac{60 \cdot \text{dakika} \cdot 0{,}93}{\Delta t} \right\rfloor$,
   $S_w = 0{,}03\,S$.

$0{,}93$ katsayısı doğrulama ve kontrol noktası yazımı için ayrılan paydır.

### Ölçümün eğitimi bozmaması

Ölçüm adımları gerçek ileri/geri geçiş ve eniyileyici çağrısı içerir. Bu
adımlar $\eta = \eta_{\max}$ ile atılsaydı — ısınmadan **önce** — model
kararsız bir bölgeye taşınırdı.

Kod, ölçüm süresince tüm parametre gruplarının $\eta$'sını sıfıra çeker ve
sonra geri yükler. $\eta = 0$ olduğunda AdamW güncellemesi

$$
\theta_t = \theta_{t-1} - 0 \cdot (\cdots) = \theta_{t-1}
$$

olur; ağırlık sönümü de $\eta$ ile çarpıldığından o da etkisizdir. Parametreler
kımıldamaz.

**Kalan etki.** Parametreler değişmese de AdamW'nin iç durumu ilerler: $m_t$,
$v_t$ momentleri güncellenir ve adım sayacı $t$ artar. Dolayısıyla gerçek
eğitimin ilk adımında sapma düzeltmesi $(1-\beta_1^{t})$ zaten 27 adım
"ısınmış" durumdadır. Etki küçüktür ve ısınma çizelgesiyle örtüşür; yine de
tam bir sıfırlama isteniyorsa ölçümden sonra eniyileyici yeniden kurulmalıdır.

**Doğrulama.** Düzeltmeden önce ilk adımdaki kayıp $\log V$'nin altında
çıkıyordu. Düzeltmeden sonra $V = 4096$ için ölçülen: $8{,}3399$, beklenen
$\log 4096 = 8{,}3178$. §1.3'teki gösterge bu hatayı doğrudan yakaladı.

---

## 3.8 Kontrol noktaları ve doğrulama

Doğrulama kümesi, belgelerin `val_fraction = 0,0005` oranıyla ($\approx$ 1/2000) ayrılmasıyla oluşturulur ve eğitimde hiç kullanılmaz.
`eval_every` adımda bir, `eval_steps` yığın üzerinde ortalama kayıp hesaplanır.

İki kontrol noktası tutulur:

- `best.pt` — en düşük doğrulama kaybının görüldüğü adım
- `last.pt` — son adım

**Ayrım ne zaman anlamlı.** Aşırı uyum (overfitting), model veriyi genellemek
yerine ezberlediğinde ortaya çıkar; belirtisi eğitim kaybı düşerken doğrulama
kaybının yükselmesidir. Bu, aynı verinin defalarca gösterildiği durumlarda
görülür.

Ön eğitimde korpus bütçeden büyüktür (§3.5) ve çoğu token bir kez bile
görülmez; ezberlenecek bir şey yoktur. Dolayısıyla doğrulama kaybı sona kadar
düşer ve `best.pt` ≈ `last.pt` olur. İkisi yine de tutulur, çünkü veri
miktarı veya epok sayısı değiştiğinde bu varsayım bozulur.

---

## 3.9 Kaynakça

- Kingma ve Ba (2014), *Adam: A Method for Stochastic Optimization*, arXiv:1412.6980.
- Loshchilov ve Hutter (2019), *Decoupled Weight Decay Regularization*, arXiv:1711.05101.
- Loshchilov ve Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv:1608.03983.
- Micikevicius ve ark. (2018), *Mixed Precision Training*, arXiv:1710.03740.
- Pascanu ve ark. (2013), *On the difficulty of training Recurrent Neural Networks*, arXiv:1211.5063 (gradyan kırpma).

---

**[← 2. Model mimarisi](02-model-architecture.md) | [İçindekiler](README.md) | Sonraki: [4. Tokenizasyon →](04-tokenization.md)**

