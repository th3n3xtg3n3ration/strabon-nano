---

**[← 1. Problem tanımı](01-problem-statement-and-objective.md) | [İçindekiler](README.md) | Sonraki: [3. Eğitim yordamı →](03-training-procedure.md)**

---

# 2. Model mimarisi

Strabon, çözücü-yalnız (decoder-only) bir dönüştürücüdür (transformer). Bu
bölümde her bileşen için önce yapılan işlem tanımlanır, sonra seçimin gerekçesi
verilir.

Genel akış:

$$
h^{(0)} = E[x]
$$

$$
u^{(\ell)} = h^{(\ell-1)} + \mathrm{Attn}\big(\mathrm{Norm}_1(h^{(\ell-1)})\big), \qquad
h^{(\ell)} = u^{(\ell)} + \mathrm{FFN}\big(\mathrm{Norm}_2(u^{(\ell)})\big)
$$

$$
z = \mathrm{Norm}_{\text{son}}(h^{(L)})\, E^{\top}
$$

İki artık adımı **ardışıktır**: ileri besleme katmanı, dikkat katmanının
güncellediği $u^{(\ell)}$ üzerinde çalışır.

Normalizasyonun alt katmandan **önce** uygulandığına dikkat edin (ön-norm /
pre-norm). Bu, artık yolunu (residual) normalizasyondan bağımsız bırakır ve
derin ağlarda eğitimi belirgin biçimde kararlı kılar.

---

## 2.1 Gömme katmanı

Token kimlikleri ayrık simgelerdir; üzerlerinde aritmetik tanımlı değildir.
Gömme matrisi $E \in \mathbb{R}^{V \times d}$ her token'a bir yoğun vektör atar:

$$
h^{(0)}_t = E[x_t] \in \mathbb{R}^{d}
$$

$E$ öğrenilen bir parametredir; $\mathcal{N}(0, 0{,}02^2)$ ile ilklenir.

**Sonuç.** Eğitim ilerledikçe dağılımsal olarak benzer token'lar $E$ uzayında
birbirine yakınsar. Bu, hedeflenmiş bir kısıt değil, amaç fonksiyonunun
dolaylı sonucudur: benzer bağlamlarda geçen token'lar benzer koşullu dağılımlar
üretmelidir, bu da benzer temsilleri gerektirir.

---

## 2.2 RMSNorm

Katmanlar üst üste bindikçe etkinleştirme (activation) büyüklükleri ya üstel
olarak büyür ya söner. Normalizasyon her alt katmanın girdisini sabit ölçeğe
çeker.

$$
\mathrm{RMSNorm}(x) \;=\; \frac{x}{\sqrt{\dfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}} \odot g
$$

$g \in \mathbb{R}^{d}$ öğrenilen kazanç vektörüdür, $\varepsilon = 10^{-5}$.

**LayerNorm ile farkı.** LayerNorm önce ortalamayı çıkarır:
$(x-\mu)/\sigma \odot g + b$. RMSNorm ortalama çıkarmaz ve kaydırma (bias)
kullanmaz; yalnızca karekök-ortalama-kare ile ölçekler.

**Gerekçe.** Zhang ve Sennrich (2019), normalizasyonun yararının ölçek
değişmezliğinden geldiğini, ortalama merkezlemenin katkısının ihmal edilebilir
olduğunu gösterdi. RMSNorm iki parametre grubundan birini ve bir geçişi ortadan
kaldırır; kalite kaybı ölçülmemiştir. Llama, Gemma ve Qwen aileleri bu nedenle
RMSNorm kullanır.

**Uygulama notu.** Karma hassasiyet (§3.6) altında karekök ve bölme fp32'de
yapılır; fp16'da $\sum x_i^2$ toplamı kolayca taşar.

---

## 2.3 Ölçekli nokta-çarpım dikkati

Her konum, geçmiş konumlardan bilgi toplar. Girdi $X \in \mathbb{R}^{T \times d}$
üç doğrusal izdüşümden geçirilir:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Dikkat ağırlıkları ve çıktı:

$$
A = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}} + M\right), \qquad
\mathrm{Attn}(X) = A V W_O
$$

Nedensel maske:

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

$-\infty$ girdileri softmax sonrasında tam sıfır ağırlık verir; böylece §1.5'teki
kısıt sağlanır. Uygulamada `F.scaled_dot_product_attention(..., is_causal=True)`
çağrılır; PyTorch uygun donanımda maskeyi hiç somutlaştırmadan FlashAttention
çekirdeğine düşer.

### Neden $\sqrt{d_h}$ ile bölünüyor

$q, k \in \mathbb{R}^{d_h}$ bileşenleri bağımsız, ortalaması 0 ve varyansı 1
olsun. İç çarpım:

$$
\mathbb{E}[q^{\top}k] = 0, \qquad
\mathrm{Var}(q^{\top}k) = \sum_{i=1}^{d_h} \mathrm{Var}(q_i k_i) = d_h
$$

Yani standart sapma $\sqrt{d_h}$'dir ve $d_h$ büyüdükçe logit'ler büyür.
Büyük logit'lerde softmax doyuma girer: çıktı neredeyse tek-noktalı olur ve
Jacobian'ı $\mathrm{diag}(p) - pp^{\top}$ sıfıra yaklaşır — **gradyan
kaybolur**. $\sqrt{d_h}$'ye bölmek varyansı 1'e sabitler ve bu doyumu önler.

**Ölçüm.** Rastgele $q, k$ ile iç çarpım varyansı:

| $d_h$ | $\mathrm{Var}(q^{\top}k)$ ölçülen | kuram | $\sqrt{d_h}$'ye bölündükten sonra |
|---|---|---|---|
| 16 | 15,97 | 16 | 0,998 |
| 64 | 64,19 | 64 | 1,003 |
| 128 | 128,40 | 128 | 1,003 |

Doyumun somut etkisi ($d_h = 64$, 32 anahtar üzerinden softmax):

| | Ortalama en büyük olasılık | Dağılımın entropisi |
|---|---|---|
| Bölme yapılmadan | 0,831 | 0,46 nat |
| $\sqrt{d_h}$ ile bölünerek | 0,165 | 3,02 nat |

Üst sınır $\log 32 = 3{,}47$ nat'tır. Bölme yapılmadığında dağılım neredeyse
tek-noktalıdır; softmax Jacobian'ı $\mathrm{diag}(p) - pp^{\top}$ bu
durumda sıfıra yakındır ve dikkat katmanı ilklemede öğrenemez.

### Çok başlıklı dikkat

Tek bir dikkat dağılımı, konumlar arası tek bir ilişki türü kodlayabilir.
İzdüşümler $H$ bağımsız **başlığa** bölünür; her başlık $d_h = d/H$ boyutunda
kendi $Q,K,V$ altuzayında çalışır, çıktılar birleştirilip $W_O$ ile karıştırılır.

$$
\mathrm{MHA}(X) = \big[\mathrm{head}_1; \dots; \mathrm{head}_H\big] W_O
$$

Toplam hesap tek başlıklı hâle eşittir ($H \cdot d_h = d$); kazanç, farklı
başlıkların farklı ilişki türlerine uzmanlaşabilmesidir. `nano-10m`'de $H = 8$.

---

## 2.4 Gruplanmış sorgu dikkati (GQA)

Çözümleme sırasında geçmiş konumların $K$ ve $V$ değerleri yeniden
hesaplanmamak üzere saklanır (KV önbelleği). Bellek maliyeti:

$$
\text{KV önbelleği} \;=\; 2 \cdot L \cdot T \cdot H_{kv} \cdot d_h \cdot (\text{bayt/sayı})
$$

($H_{kv}$ anahtar-değer başlığı sayısıdır; GQA kapalıyken $H_{kv} = H$.)

Uzun bağlamda bu terim model ağırlıklarını geçebilir.

GQA, sorgu başlığı sayısını korurken anahtar-değer başlığı sayısını
$H_{kv} < H$'ye düşürür; her $K,V$ çifti $H/H_{kv}$ sorgu başlığı tarafından
paylaşılır. `mini-500m`'de $H = 16$, $H_{kv} = 4$ olduğundan önbellek **dörtte
bire** iner.

**Bedeli.** Ainslie ve ark. (2023), makul oranlarda kalite kaybının küçük
olduğunu bildirir. `nano-*` yapılandırmalarında $H_{kv} = H$'dir (GQA kapalı),
çünkü bu ölçekte KV önbelleği zaten sorun değildir.

---

## 2.5 Döndürmeli konum kodlaması (RoPE)

### Sorun

Dikkat işlemi konum bilgisinden bağımsızdır: $A$ hesabında $t$ indisi hiçbir
yerde geçmez. Girdi sırası değiştirilirse çıktı da aynı biçimde yer değiştirir
(permütasyon eşdeğerliği). Dolayısıyla "kedi köpeği kovaladı" ile "köpek
kediyi kovaladı" model için ayırt edilemez. Konum bilgisi dışarıdan verilmelidir.

### Yöntem

RoPE, $Q$ ve $K$ vektörlerini konuma bağlı bir açıyla döndürür. $d_h$ boyutu
$d_h/2$ çifte ayrılır; $i$. çift için taban frekans:

$$
\theta_i = b^{-2i/d_h}, \qquad b = 10^4, \quad i = 0, \dots, \tfrac{d_h}{2}-1
$$

$m$ konumundaki dönüş, her çift üzerine uygulanan 2×2 döndürme matrisidir:

$$
R_{m,i} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
$$

Bloklar köşegen olarak birleştirilerek $R_m \in \mathbb{R}^{d_h \times d_h}$
elde edilir ve uygulanır: $\tilde q_m = R_m q_m$, $\tilde k_n = R_n k_n$.

### Neden işe yarıyor: göreli konum özelliği

Döndürme matrisleri için $R_a^{\top} = R_{-a}$ ve $R_a R_b = R_{a+b}$ geçerlidir.
Dikkat skoru:

$$
\tilde q_m^{\top} \tilde k_n
= (R_m q_m)^{\top} (R_n k_n)
= q_m^{\top} R_m^{\top} R_n k_n
= q_m^{\top} R_{\,n-m}\, k_n
$$

**Skor yalnızca $n-m$ farkına bağlıdır.** Model mutlak konumu değil göreli
uzaklığı görür. Bunun iki sonucu vardır:

1. 3.→5. konum ilişkisi ile 103.→105. konum ilişkisi aynı biçimde kodlanır;
   model tek bir kural öğrenir.
2. Eğitimde görülmemiş uzunluklara genelleme, öğrenilmiş mutlak konum
   gömmelerine göre belirgin biçimde daha iyidir.

**Sayısal doğrulama.** Rastgele $q, k \in \mathbb{R}^{16}$ ile ölçülen değerler:

| $m$ | $n$ | $n-m$ | $\tilde q_m^{\top}\tilde k_n$ | $q^{\top}R_{n-m}k$ | fark |
|---|---|---|---|---|---|
| 3 | 5 | 2 | −3,786527849631 | −3,786527849631 | $1{,}8\cdot10^{-15}$ |
| 103 | 105 | 2 | −3,786527849631 | −3,786527849631 | $1{,}8\cdot10^{-15}$ |
| 0 | 7 | 7 | −6,986334445552 | −6,986334445552 | 0 |
| 20 | 27 | 7 | −6,986334445552 | −6,986334445552 | 0 |
| 100 | 103 | 3 | −4,810435694306 | −4,810435694306 | 0 |
| 200 | 203 | 3 | −4,810435694306 | −4,810435694306 | $2{,}7\cdot10^{-15}$ |

Aynı farka sahip konum çiftleri, mutlak konumları 100 veya 200 birim ayrı olsa
bile **aynı** skoru üretir.

> **Uygulama ayrıntısı.** Açı tabloları `float64`'te biriktirilip sonra
> indirgenir. $m\theta_i$ çarpımı konumla büyüdüğünden `float32`'de hesaplamak
> uzun bağlamda birkaç anlamlı basamak yitirir; özdeşlik o durumda $10^{-15}$
> yerine yalnızca $10^{-6}$ mertebesinde sağlanır. Tablolar bir kez
> kurulduğundan ek duyarlığın maliyeti yoktur.

Ek olarak $R_m$ ortogonaldir, dolayısıyla $\lVert R_m q \rVert = \lVert q \rVert$:
dönüşüm hiçbir bilgi yok etmez, yalnızca yönlendirir. `tests/test_model.py`
bu özelliği doğrular (ölçülen en büyük norm sapması: $4{,}77 \times 10^{-7}$).

**Uygulama notu.** Kod, $i$. boyutu $(i, i + d_h/2)$ ile eşleştiren `rotate_half`
biçimini kullanır; kuramsal gösterim $(2i, 2i{+}1)$ eşleştirmesidir. İkisi
boyutların bir permütasyonuyla birbirine denktir ve $W_Q, W_K$ öğrenildiği için
fark yaratmaz.

---

## 2.6 SwiGLU ileri besleme katmanı

Dikkat, konumlar **arasında** bilgi taşır. İleri besleme katmanı ise her konum
üzerinde **bağımsız** olarak doğrusal olmayan bir dönüşüm uygular.

Klasik dönüştürücüde:

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(xW_1)W_2, \qquad f = 4d
$$

Strabon'da kapılı (gated) sürüm kullanılır:

$$
\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_1) \odot xW_3\big) W_2,
\qquad \mathrm{SiLU}(u) = u \cdot \sigma(u) = \frac{u}{1+e^{-u}}
$$

$W_3$ bir **kapı** görevi görür: $\mathrm{SiLU}(xW_1)$ çıktısı, girdiye bağlı
bir çarpanla ölçeklenir. Böylece katmanın etkin davranışı girdiye göre
değişebilir; sabit bir doğrusal olmayanlığa göre daha esnektir.

### Neden $f = 8d/3$

SwiGLU iki yerine üç matris kullanır. Parametre sayısını klasik katmanla eşit
tutmak için:

$$
3df = 2 \cdot d \cdot 4d \;\;\Longrightarrow\;\; f = \frac{8d}{3}
$$

Kod bu değeri 64'ün katına yukarı yuvarlar (tensör çekirdeklerinde hizalama
için). $d = 256$ için $8 \cdot 256/3 = 682{,}7 \to f = 704$.

**Gerekçe.** Shazeer (2020), eşit parametre bütçesinde kapılı
değişkelerin ReLU tabanlı katmanı tutarlı biçimde geçtiğini bildirir. Llama,
PaLM ve Qwen aileleri bu nedenle SwiGLU kullanır.

---

## 2.7 Artık bağlantılar ve ilkleme ölçeği

Her alt katman çıktısını girdisinin **üzerine ekler**:

$$
h^{(\ell)} = h^{(\ell-1)} + F\big(\mathrm{Norm}(h^{(\ell-1)})\big)
$$

Gerekçe gradyan akışıdır. Türev alındığında:

$$
\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} = I + \frac{\partial F}{\partial h^{(\ell-1)}}
$$

Birim matris terimi, gradyanın $L$ katman boyunca çarpımlarla sönmesini önleyen
doğrudan bir yol açar. Bu olmadan derin ağlar eğitilemez.

### Ölçek sorunu ve $1/\sqrt{2L}$

Her alt katman varyansı $\sigma^2$ olan bir terim ekliyorsa, $L$ blokta $2L$
alt katman bulunduğundan artık akışın varyansı yaklaşık $2L\sigma^2$'ye çıkar.
Derinlik arttıkça girdi büyüklüğü kontrolsüz büyür.

Çözüm, artık yola yazan izdüşümleri ($W_O$ ve $W_2$) küçültülmüş ölçekle
ilklemektir:

$$
W \sim \mathcal{N}\!\left(0, \left(\frac{0{,}02}{\sqrt{2L}}\right)^{2}\right)
$$

Böylece toplam varyans derinlikten bağımsız olarak $O(1)$ kalır. GPT-2 ile
tanıtılan bu düzeltme kodda `model.py` içinde yalnızca `wo.weight` ve
`w2.weight` parametrelerine uygulanır.

---

## 2.8 Ağırlık bağlama

Çıkış katmanı ayrı bir matris kullanmaz; gömme matrisinin devriği alınır:

$$
z = \mathrm{Norm}(h^{(L)})\, E^{\top}
$$

**Gerekçe.** İki matris de "token ile temsil arasındaki ilişki"yi kodlar; biri
ileri, diğeri ters yönde. Press ve Wolf (2017), bağlamanın parametre
tasarrufunun yanı sıra küçük modellerde şaşkınlığı **iyileştirdiğini** gösterdi.

**Ölçülen etki.** `nano-10m` için:

| | Parametre |
|---|---|
| Gömme matrisi $E$ | 4.194.304 |
| Gövde (katmanlar + son norm) | 4.820.224 |
| **Bağlı toplam** | **9.014.528** |
| Bağlanmasaydı | 13.208.832 (%47 daha büyük) |

**Sabit $V$ için** gömme payı model küçüldükçe artar; bu nedenle bağlama küçük
modellerde neredeyse her zaman doğru seçimdir. Aşağıdaki tabloda `nano-50m`'in
payı `nano-25m`'inkinden yüksektir çünkü sözlük de büyütülmüştür.

| Yapılandırma | $V$ | $d$ | Gömme payı |
|---|---|---|---|
| `debug` | 4.096 | 128 | %55 |
| `nano-10m` | 16.384 | 256 | %47 |
| `nano-25m` | 16.384 | 384 | %26 |
| `nano-50m` | 32.768 | 512 | %34 |
| `mini-500m` | 32.768 | 1.280 | %8 |

---

## 2.9 Kaydırma terimlerinin (bias) kaldırılması

Hiçbir doğrusal katmanda kaydırma terimi kullanılmaz. Ön-norm mimarisinde
RMSNorm'un öğrenilen kazancı zaten bir serbestlik derecesi sağlar; kaydırma
terimlerinin ek katkısı büyük ölçekli çalışmalarda ölçülememiştir (Chowdhery
ve ark., 2022). Parametre ve bellek trafiği tasarrufu sağlar.

---

## 2.10 Parametre sayımı

Kapalı biçim:

$$
\begin{aligned}
N_{\text{gömme}} &= Vd \\
N_{\text{dikkat}} &= \underbrace{d^2}_{W_Q} + \underbrace{2\,d\,H_{kv}d_h}_{W_K, W_V} + \underbrace{d^2}_{W_O} \\
N_{\text{ffn}} &= 3df \\
N_{\text{katman}} &= N_{\text{dikkat}} + N_{\text{ffn}} + 2d \\
N &= Vd + L\,N_{\text{katman}} + d
\end{aligned}
$$

Son $+2d$ ve $+d$ terimleri RMSNorm kazançlarıdır. `config.py` bu formülü
uygular ve `tests/test_model.py` formülün gerçek model ile birebir uyuştuğunu
doğrular — mimaride sessiz bir değişiklik olursa sınama düşer.

| Ön ayar | $V$ | $L$ | $H/H_{kv}$ | $d$ | $f$ | $T$ | $N$ |
|---|---|---|---|---|---|---|---|
| `debug` | 4.096 | 2 | 4 | 128 | 384 | 256 | 1,0 M |
| `nano-10m` | 16.384 | 6 | 8 | 256 | 704 | 512 | 9,0 M |
| `nano-25m` | 16.384 | 10 | 8 | 384 | 1.024 | 512 | 24,0 M |
| `nano-50m` | 32.768 | 10 | 8 | 512 | 1.408 | 1.024 | 48,9 M |
| `mini-500m` | 32.768 | 26 | 16/4 | 1.280 | 3.456 | 2.048 | 493,6 M |

---

## 2.11 Hesap maliyeti

Token başına ileri + geri geçiş için yaygın kestirim:

$$
C_{\text{token}} \;\approx\; 6N \;+\; 12\,L\,H\,d_h\,T
$$

İlk terim: ileri geçişte parametre başına bir çarpma ve bir toplama ($2N$),
geri geçiş bunun iki katı ($4N$). İkinci terim, dikkatin $T$ ile doğrusal
büyüyen ek maliyetidir ve uzun bağlamda baskın hâle gelir.

Bu kestirim §5.1'deki ölçekleme hesaplarında kullanılır. (Eğitim betiği FLOP
hesabı yapmaz; yalnızca saniyedeki token sayısını ve token/parametre oranını
raporlar.)

**Dikkat terimi ne zaman baskın olur?** İki terim $6N = 12LHd_hT = 12LdT$
noktasında eşitlenir. `nano-10m` için ($N = 9{,}01$ M, $L = 6$, $d = 256$) bu
$T \approx 2\,900$'e karşılık gelir. Varsayılan $T = 512$'de dikkat terimi
toplamın yaklaşık %15'idir; bağlam dört katına çıkarıldığında pay yarıya
yaklaşır.

---

## 2.12 Kaynakça

- Vaswani ve ark. (2017), *Attention Is All You Need*, arXiv:1706.03762.
- Zhang ve Sennrich (2019), *Root Mean Square Layer Normalization*, arXiv:1910.07467.
- Su ve ark. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Shazeer (2020), *GLU Variants Improve Transformer*, arXiv:2002.05202.
- Ainslie ve ark. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245.
- Press ve Wolf (2017), *Using the Output Embedding to Improve Language Models*, arXiv:1608.05859.
- Xiong ve ark. (2020), *On Layer Normalization in the Transformer Architecture*, arXiv:2002.04745.
- Chowdhery ve ark. (2022), *PaLM: Scaling Language Modeling with Pathways*, arXiv:2204.02311.

*Yıl bilgisi, aksi belirtilmedikçe arXiv ilk sürüm yılıdır.*

---

**[← 1. Problem tanımı](01-problem-statement-and-objective.md) | [İçindekiler](README.md) | Sonraki: [3. Eğitim yordamı →](03-training-procedure.md)**

