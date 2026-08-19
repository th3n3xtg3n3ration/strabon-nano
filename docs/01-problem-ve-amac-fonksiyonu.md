# 1. Problem tanımı ve amaç fonksiyonu

## 1.1 Kapsam

Bu belge dizisi, Strabon Nano'nun kuramsal temelini ve uygulama ayrıntılarını
tanımlar. Amaç, modeldeki her bileşenin *hangi matematiksel işlemi yaptığını*,
*neden o işlemin seçildiğini* ve *seçimin ölçülebilir sonucunu* kayda geçirmektir.

Ön koşul: doğrusal cebir (matris çarpımı, iç çarpım), çok değişkenli türev ve
temel olasılık. Sinir ağları hakkında ön bilgi varsayılmaz.

**Notasyon.** Belge boyunca:

| Sembol | Anlam |
|---|---|
| $V$ | sözlük büyüklüğü (token sayısı) |
| $T$ | dizi uzunluğu (bağlam penceresi) |
| $B$ | yığın büyüklüğü |
| $d$ | model boyutu (`d_model`) |
| $L$ | katman sayısı |
| $H$ | dikkat başlığı sayısı |
| $d_h = d/H$ | başlık boyutu |
| $f$ | ileri besleme ara boyutu (`d_ff`) |
| $N$ | toplam parametre sayısı |
| $D$ | eğitim token sayısı |
| $x_t$ | dizideki $t$. token |
| $\theta$ | modelin tüm parametreleri |

---

## 1.2 Otoregresif dil modelleme

Bir dil modeli, token dizileri üzerinde bir olasılık dağılımı $p_\theta(x)$
öğrenir. Zincir kuralı gereği herhangi bir ortak dağılım koşullu olasılıkların
çarpımına ayrıştırılabilir:

$$
p_\theta(x_1, \dots, x_T) \;=\; \prod_{t=1}^{T} p_\theta\!\left(x_t \mid x_{<t}\right)
$$

Bu ayrıştırma yaklaşık değildir; her zaman geçerlidir. Sonucu şudur: dizi
üzerindeki dağılımı modellemek, **tek bir koşullu dağılımı** — "geçmişe bakarak
bir sonraki token" — modellemeye indirgenir.

Modelin görevi bu koşullu dağılımı üretmektir. Her konum için $V$ boyutlu bir
skor (logit) vektörü $z_t \in \mathbb{R}^V$ hesaplar ve bunu softmax ile
olasılığa çevirir:

$$
p_\theta(x_t = j \mid x_{<t}) \;=\; \operatorname{softmax}(z_t)_j
\;=\; \frac{\exp(z_{t,j})}{\sum_{k=1}^{V} \exp(z_{t,k})}
$$

Softmax'ın işlevi, sınırsız gerçel skorları toplamı 1 olan pozitif sayılara
dönüştürmektir. Üstel fonksiyon seçimi keyfi değildir: softmax,

$$
\operatorname{softmax}(z) = \arg\max_{p \in \Delta^{V-1}} \Big\{ \textstyle\sum_i p_i z_i + H(p) \Big\},
\qquad H(p) = -\sum_i p_i \log p_i
$$

yani beklenen skor ile entropinin toplamını enbüyüten dağılımdır ($\Delta^{V-1}$
olasılık yalınlığı). Ayrıca türevi (§1.4) özellikle sade bir biçim alır.

### Neden bu görev yeterli

"Bir sonraki token" görevi yüzeysel görünür. Ancak koşullu dağılımı *iyi*
kestirmek, dizideki bilginin tamamını kullanmayı gerektirir:

| Dizi | Doğru kestirim için gereken |
|---|---|
| *Türkiye'nin başkenti \_\_\_* | olgusal bilgi |
| *2 + 2 = \_\_\_* | aritmetik |
| *Ali kitabı Ayşe'ye verdi. O \_\_\_* | gönderim çözümleme (anafora) |

Dil modellerinde gözlemlenen yetenekler bu tek görevin yan ürünüdür. Küçük
modeller görevin yalnızca en yüzeysel katmanını — biçimbirimsel ve sözdizimsel
düzenlilikleri — öğrenebilir.

---

## 1.3 Amaç fonksiyonu: negatif log-olabilirlik

Eğitim, verinin model altındaki olabilirliğini enbüyütmektir. Sayısal
kararlılık ve toplamsallık için logaritma alınır ve işaret ters çevrilerek bir
enküçültme problemine dönüştürülür:

$$
\mathcal{L}(\theta) \;=\; -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta\!\left(x^{(b)}_t \mid x^{(b)}_{<t}\right)
$$

Bu, hedef dağılım tek-noktalı (one-hot) olduğunda **çapraz entropi** ile
özdeştir. Uygulamada logit'lerden doğrudan hesaplanır:

$$
-\log \operatorname{softmax}(z)_{y} \;=\; \log\!\sum_{k} \exp(z_k) \;-\; z_y
$$

Sağ taraf `log-sum-exp` biçimidir ve taşma yaşamamak için
$\log\sum_k \exp(z_k - \max_j z_j) + \max_j z_j$ olarak hesaplanır. PyTorch'un
`F.cross_entropy` çağrısı bunu içeride yapar; kodda `model.py` içinde
kullanılan budur.

### Kaybın yorumu

$\mathcal{L}$ birimi *nat*'tır (doğal logaritma tabanında bilgi). İki türetilmiş
büyüklük kullanılır:

**Şaşkınlık (perplexity).**

$$
\mathrm{PPL} \;=\; \exp(\mathcal{L})
$$

Yorumu: model, her konumda eşit olasılıklı $\mathrm{PPL}$ seçenek arasında
kararsız kalıyormuş gibi davranmaktadır. $\mathcal{L}=0{,}84$ için
$\mathrm{PPL}=2{,}32$; yani model ortalama iki-üç aday arasında bocalamaktadır.

**Rastgele başlangıç değeri.** Eğitilmemiş bir modelde logit'ler yaklaşık
sıfır etrafında ve birbirinden bağımsızdır, dolayısıyla softmax çıktısı düzgün
dağılıma yakındır. O durumda:

$$
\mathcal{L}_0 \;\approx\; -\log \frac{1}{V} \;=\; \log V
$$

Bu, uygulamanın doğruluğunu sınamak için **en ucuz ve en güvenilir** göstergedir.
Ölçülen değerler:

| $V$ | $\log V$ | Ölçülen $\mathcal{L}_0$ (5 tohum) | Ortalama |
|---|---|---|---|
| 512 | 6,2383 | 6,2380 – 6,2896 | 6,2611 |
| 4.096 | 8,3178 | 8,3149 – 8,3579 | 8,3344 |
| 16.384 | 9,7041 | 9,6845 – 9,7380 | 9,7095 |

**Alt sınır.** Log-sum-exp dışbükey olduğundan, Jensen eşitsizliği gereği
$\mathbb{E}[\mathrm{LSE}(z)] \geq \mathrm{LSE}(\mathbb{E}[z]) = \mathrm{LSE}(0) = \log V$.
Logit'lerin ortalaması sıfır olduğundan $\mathbb{E}[\mathcal{L}_0] \geq \log V$
olmalıdır. Tek tek ölçümler sonlu örneklem nedeniyle bu değerin bir miktar
altına inebilir, ancak sistemli olarak altında kalan bir sonuç hatayı gösterir.

Sapmanın olağan nedenleri ilkleme ölçeği hatası veya hedef sızıntısıdır.
`tests/test_model.py` bu sınamayı otomatik yapar.

> **Uygulama tuzağı.** Bu sınamada hedefler girdilerden **bağımsız** seçilmelidir.
> Ağırlık bağlama (§2.8) nedeniyle model, girdi token'ını geri kestirme
> görevini ilklemede zaten kısmen çözer; hedef olarak girdiyi vermek kaybı
> yapay biçimde düşürür ve hatayı gizler.

---

## 1.4 Gradyan

Eğitim, $\nabla_\theta \mathcal{L}$ yönünde iniş yapmaktır. Zincirin ilk halkası
— kaybın logit'lere göre türevi — kapalı biçimde ve çarpıcı biçimde sadedir.
$p = \operatorname{softmax}(z)$ ve $y$ tek-noktalı hedef olmak üzere:

$$
\frac{\partial \mathcal{L}}{\partial z} \;=\; p - y
$$

Yani gradyan, **kestirilen dağılım ile gerçek dağılım arasındaki farktır**.
(PyTorch'un otomatik türevi ile bu kapalı biçim arasındaki en büyük sapma
`float64` duyarlıkta $2{,}8 \times 10^{-17}$ ölçülmüştür — makine hassasiyeti
mertebesinde.)
Model doğru token'a olasılık 1 verirse gradyan sıfırlanır. Softmax ve çapraz
entropinin birlikte kullanılmasının nedeni budur: ayrı ayrı ele alındığında
ortaya çıkan üstel terimler birbirini götürür.

Geri kalan türevler geri yayılım (backpropagation) ile hesaplanır; bu, zincir
kuralının hesap çizgesi üzerinde ters sırada uygulanmasıdır. PyTorch bunu
otomatik yapar.

---

## 1.5 Nedensellik kısıtı

$p_\theta(x_t \mid x_{<t})$ tanımı gereği model, $t$ konumundaki kestirimi
yaparken $x_{\geq t}$'yi **görmemelidir**. Bu kısıt dikkat mekanizmasında bir
maske ile uygulanır (§2.3).

Kısıt ihlal edilirse eğitim kaybı hızla düşer — model cevabı girdiden okur —
fakat çözümleme (decoding) sırasında gelecek konumlar tanımsız olduğundan model
kullanılamaz hâle gelir. Bu hata çökmeye yol açmadığı ve kayıp eğrisini
*iyileştirdiği* için sınamasız fark edilmesi olası değildir.

`tests/test_model.py` içindeki sınama, $t \geq 10$ konumlarındaki token'ları
değiştirip $t < 10$ çıktılarının değişmediğini doğrular. Ölçülen fark: önek
$0{,}0$, sonek $1{,}27$.

---

## 1.6 Kaynakça

- Bengio ve ark. (2003), *A Neural Probabilistic Language Model*.
- Vaswani ve ark. (2017), *Attention Is All You Need*, arXiv:1706.03762.
- Radford ve ark. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2).

---

**Sonraki:** [2. Model mimarisi](02-model-mimarisi.md)
