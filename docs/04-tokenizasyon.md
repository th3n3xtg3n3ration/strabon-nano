# 4. Tokenizasyon ve Türkçe'nin getirdiği kısıtlar

## 4.1 Neden token

Model, ayrık simgeler üzerinde çalışır. İki uç seçenek de elverişsizdir:

**Karakter düzeyi.** Sözlük küçüktür ve bilinmeyen simge sorunu yoktur, ancak
diziler çok uzar. Dikkatin maliyeti $T$ ile büyüdüğünden (§2.11) bu doğrudan
hesap maliyetidir. Ayrıca model her sözcüğü karakterlerden yeniden kurmak
zorunda kalır.

**Sözcük düzeyi.** Diziler kısadır, ancak sözlük denetlenemez biçimde büyür.
Türkçe gibi eklemeli bir dilde bu özellikle ağırdır: tek bir kökten üretilebilen
biçim sayısı, ek dizilerinin bileşimi kadar çoktur.

> *ev* → *evde* → *evimde* → *evlerimizdekilerden*

Bu biçimlerin her biri ayrı bir sözlük girdisi olsaydı, aralarındaki
ilişki modele hiçbir şekilde görünmezdi.

**Alt-sözcük düzeyi** bu ikisinin arasındadır ve standart çözümdür.

---

## 4.2 Bayt çifti kodlaması (BPE)

### Algoritma

Korpus, sözcük **türleri** ve frekanslarından oluşan bir çoklu küme olarak ele
alınır. Her tür, temel alfabedeki simgelerin dizisi olarak yazılır.

1. $\mathcal{V} \leftarrow$ temel alfabe (256 bayt)
2. Aşağıdakini $|\mathcal{V}|$ hedef büyüklüğe ulaşana dek yinele:
   a. Tüm dizilerde bitişik $(a,b)$ çiftlerinin frekansını hesapla:
      $\;c(a,b) = \sum_{w} \mathrm{freq}(w)\cdot \#\{(a,b) \in w\}$
   b. $(a^\*, b^\*) = \arg\max c(a,b)$
   c. Tüm dizilerde $(a^\*,b^\*)$ örüntüsünü tek simge $a^\*b^\*$ ile değiştir
   d. $\mathcal{V} \leftarrow \mathcal{V} \cup \{a^\*b^\*\}$ ve birleştirmeyi kaydet

Kodlama sırasında aynı birleştirmeler **öğrenildikleri sırayla** uygulanır.

### Bayt düzeyi olmasının sonucu

Temel alfabe karakterler değil **baytlardır**. İki sonucu vardır:

1. **Bilinmeyen simge yoktur.** Her girdi, tanım gereği bayt dizisidir;
   dolayısıyla her metin kodlanabilir. Emoji, farklı yazı sistemleri, bozuk
   kodlama — hepsi temsil edilir.
2. **Türkçe harfler iki bayttır.** UTF-8'de ASCII dışı karakterler çok bayt
   kaplar; ç, ğ, ı, ö, ş, ü ve büyük harfli karşılıkları 2 bayttır. Model,
   "ç" simgesini bile iki bayttan birleştirmeyi öğrenmek zorundadır. Bu, ilk
   birleştirmelerin bir kısmının Türkçe harfleri kurmaya harcanması demektir.

### Ön-parçalama

BPE'ye ham metin verilmez; önce kaba bir bölme uygulanır. Amaç, birleştirmelerin
sözcük sınırlarını aşmasını engellemektir — aksi hâlde " ve bir" gibi
sözcükler-arası öbekler tek token hâline gelir ve sözlük savurgan kullanılır.

Strabon'un deseni iki noktada standart GPT-2 deseninden ayrılır:

**1. Kesme işaretli ekler bütün tutulur.**

Türkçe'de özel adlara gelen çekim ekleri kesme işaretiyle ayrılır. Ölçülen
davranış:

```
metin        : bahçede Zeynep'in ağaçların 2023 yılında

GPT-2 deseni : ['bahçede', ' Zeynep', "'", 'in', ' ağaçların', ' 2023', ' yılında']
Strabon      : ['bahçede', ' Zeynep', "'in", ' ağaçların', ' ', '202', '3', ' yılında']
```

GPT-2 deseninde `'` ve `in` ayrı parçalardır. Ek tek bir birim olarak
tanınmadığında, *Zeynep'in*, *Zeynep'e*, *Zeynep'ten* biçimleri arasındaki
ortaklık modele daha zor görünür.

> **Yaygın bir yanılgı.** "GPT-2 deseni Türkçe harfleri tanımaz" iddiası
> **yanlıştır**. Desen `\p{L}` (Unicode harf sınıfı) kullanır ve ç, ğ, ı, ö, ş,
> ü bu sınıfın içindedir; yukarıdaki çıktıda *bahçede* her iki desende de tek
> parçadır. Bu belgenin önceki bir sürümünde bu hata bulunuyordu ve ölçümle
> düzeltildi.

**2. Sayı dizileri en fazla üçlü öbeklere ayrılır.** Bu, uzun sayıların sözlükte
tek tek yer kaplamasını engeller; GPT-4'ün deseninde de bulunan bir düzenlemedir.

---

## 4.3 Ölçütler

### Doğurganlık (fertility)

$$
F = \frac{\#\text{token}}{\#\text{sözcük}}
$$

Sıkıştırma verimini ölçer. Aynı metin için ölçülen değerler
(arXiv:2508.13058; 198.193 sözcüklük Türkçe korpus):

| Tokenizer | $|\mathcal{V}|$ | Token | $F$ | %TR |
|---|---|---|---|---|
| aya-expanse | 255.029 | 434.526 | 2,19 | 50,67 |
| llama-3.1 | 128.256 | 488.535 | 2,46 | 45,80 |
| gemma-2 | 256.000 | 497.015 | 2,51 | 48,63 |
| Qwen2.5 | 151.665 | 561.866 | 2,83 | 40,33 |

Kaynak yalnızca sözlük büyüklüğü, token sayısı ve %TR bildirir; $F$ sütunu
token sayısının 198.193'e bölünmesiyle **türetilmiştir**.

Karşılaştırma için İngilizce'de $F \approx 1{,}2\text{–}1{,}3$ bildirilir; bu
değer yukarıdaki tablonun kaynağından değil, ayrı ölçümlerdendir.

**Doğurganlığın maliyeti.** $F$ iki katına çıkarsa aynı içerik iki kat token
tüketir. Bu:

- eğitim maliyetini iki katına çıkarır (maliyet token sayısıyla doğrusaldır),
- etkin bağlam penceresini yarıya indirir ($T$ token sabitse, sığan metin
  yarıya iner),
- çıkarım (inference) ücretini iki katına çıkarır.

### %TR

$$
\%\mathrm{TR} = \frac{\big|\{\,v \in \mathcal{V} : \text{decode}(v)\text{'nin tüm karakterleri Türk alfabesinde}\,\}\big|}{|\mathcal{V}|} \times 100
$$

Sözlüğün ne kadarının hedef dile ayrılmış olduğunu ölçer.

**Neden bu ölçüt.** arXiv:2508.13058, dört tokenizer üzerinde %TR ile MMLU
başarımı arasında $r = 0{,}90$ ilişki bildirir; aynı çalışmada doğurganlık ile
ilişki daha zayıftır. Yorum: bir tokenizer'ın değeri yalnızca ne kadar
sıkıştırdığında değil, sözlüğünün hedef dille ne kadar **hizalı** olduğundadır.

> **İki çekince.** (i) Bu çalışmanın %TR tanımı, token'ın hedef dilde geçerli
> bir **sözcük** olup olmadığına bakar ve sözlük gerektirir; buradaki uygulama
> yalnızca karakter kümesini denetler — daha ucuz bir vekil ölçüttür,
> yayımlanan sayının yeniden üretimi değildir. (ii) $r = 0{,}90$ dört
> tokenizer ve farklı model aileleri üzerinden hesaplanmıştır; ilişki gösterir,
> nedensellik göstermez.

---

## 4.4 Türkçe'nin dilbilimsel özellikleri ve tokenizasyona etkisi

**Eklemeli (agglutinative) yapı.** Dilbilgisel ilişkiler, köke sırayla eklenen
biçimbirimlerle (morfem) kurulur. Her ek tek bir işlev taşır ve sınır belirgindir:

$$
\underbrace{ev}_{\text{kök}} + \underbrace{ler}_{\text{çoğul}} + \underbrace{imiz}_{\text{iyelik}} + \underbrace{de}_{\text{bulunma}} + \underbrace{ki}_{\text{aitlik}} + \underbrace{ler}_{\text{çoğul}} + \underbrace{den}_{\text{ayrılma}}
$$

Bu yapı BPE için **elverişlidir**: ekler yüksek frekanslı olduğundan
birleştirme sırasında erken keşfedilir. Türkçe korpusta eğitilen bir BPE,
`ler`, `lar`, `den`, `dan`, `miş`, `ecek` gibi parçaları hiçbir dilbilgisi
kuralı verilmeden bulur.

Sorun, tokenizer **başka bir dilde** eğitildiğinde ortaya çıkar. İngilizce
korpusta `tion`, `ment`, `ing` sık geçtiği için sözlüğe girer; Türkçe ekler
girmez. Sonuç, biçimbirim sınırlarıyla örtüşmeyen bölünmelerdir:

```
Hizalı bölünme    : ev | ler | imiz | den
Hizasız bölünme   : ev | ler | im | i | z | d | en
```

İkincisi hem daha uzundur hem de modele daha zor bir görev verir: `imiz`
biçimbiriminin işlevini, dört ayrı parçanın bileşiminden çıkarması gerekir.

**Ünlü uyumu.** Eklerin ünlüleri köke göre değişir: *ev-de* fakat *okul-da*;
*ev-ler* fakat *okul-lar*. Bu, her ekin 2 veya 4 yüzey biçimi olması demektir
ve sözlükte bunların hepsine yer gerekir. Modelin öğrendiği şey — hangi biçimin
hangi kökle geldiği — veri tarafından belirlenir; kural olarak verilmez.

---

## 4.5 Veri hattı

### Kaynaklar

| Kaynak | İçerik | Lisans |
|---|---|---|
| `HuggingFaceFW/fineweb-2` (`tur_Latn`) | Süzülmüş Türkçe web metni | ODC-By 1.0 |
| `wikimedia/wikipedia` (`20231101.tr`) | Türkçe Vikipedi | CC BY-SA 4.0 |

Varsayılan karışım belge sayısına göre %80 web / %20 Vikipedi'dir (400.000 /
100.000). Vikipedi maddeleri ortalama olarak daha uzun olduğundan **token
cinsinden** oran Vikipedi lehine daha yüksektir.

**Gerekçe.** Tek kaynaklı korpusların bilinen kusurları vardır: yalnızca
ansiklopedik metin tek bir yazı kaydı öğretir; yalnızca web metni gürültü
taşır. Açık eğitim tarifleri (SmolLM3, OLMo 3) aynı nedenle çok kaynaklı
karışım kullanır ve karışım oranlarını küçük ölçekli denemelerle ayarlar.

**Lisans kısıtı.** CC BY-NC-SA lisanslı Türkçe korpuslar (örn. `vngrs-web-corpus`)
bilinçli olarak dışarıda bırakılmıştır; "NC" ticari kullanımı engeller ve
eğitim tamamlandıktan sonra veri kaynağı geriye dönük değiştirilemez.

### Süzgeçler

Her belge $x$ için, aşağıdaki ölçütlerin tamamı sağlanmalıdır:

| Ölçüt | Eşik | Amaç |
|---|---|---|
| $\lvert x \rvert$ | $\geq 300$ karakter | kırıntı belgeleri ele |
| Türk alfabesi oranı | $\geq 0{,}80$ | yabancı yazı sistemlerini ele |
| Türkçe'ye özgü harf oranı | $\geq 0{,}03$ | Latin alfabeli diğer dilleri ele |
| En sık satırın payı | $\leq 0{,}30$ | menü/şablon metnini ele |
| Harf / karakter oranı | $\geq 0{,}55$ | tablo ve liste döküntüsünü ele |
| Parmak izi | görülmemiş | yinelenen belgeleri ele |

**Üçüncü ölçüt neden gerekli.** Türk alfabesi, q, w, x dışında İngiliz
alfabesini kapsar. Dolayısıyla ikinci ölçüt tek başına İngilizce metni
**elemez**: "This is an English document" dizisinin harflerinin tamamı Türk
alfabesindedir.

Türkçe'ye özgü harf kümesi $\{$ç, ğ, ı, ö, ş, ü$\}$ üzerinden ölçülen oranlar:

| Metin türü | Oran |
|---|---|
| Türkçe | %10–16 |
| Azerbaycan Türkçesi | ~%5 |
| İngilizce | %0–1 |

$0{,}03$ eşiği İngilizceyi güvenli bir payla eler. Bu ölçüt, süzgecin
İngilizce bir örneği kabul ettiği gözlemlendikten **sonra** eklenmiştir;
`tests/test_data.py` artık bu durumu sınar.

**Tekilleştirmenin sınırı.** Parmak izi, belgenin normalleştirilmiş ilk 2.000
karakterinden alınır ve küme her kaynak için ayrı tutulur; kaynaklar arası
yinelemeler yakalanmaz. Yaklaşık yineleme tespiti (MinHash vb.) bu ölçekte
uygulanmamıştır.

### Depolama biçimi

Tokenize edilmiş korpus, düz bir `uint16` dizisi olarak yazılır. Ayrıştırma
maliyeti ortadan kalkar ve `np.memmap` ile rastgele erişim mümkün olur (§3.5).

---

## 4.6 Kaynakça

- Sennrich ve ark. (2016), *Neural Machine Translation of Rare Words with Subword Units*, arXiv:1508.07909.
- Radford ve ark. (2019), GPT-2 (bayt düzeyi BPE ve ön-parçalama deseni).
- Bayram ve ark. (2025), *Tokenization Standards for Linguistic Integrity: Turkish as a Benchmark*, arXiv:2502.07057 / 2508.13058.
- Penedo ve ark. (2024), *FineWeb-2*.

---

**Sonraki:** [5. Ölçekleme, çözümleme ve doğrulama](05-olcekleme-ve-dogrulama.md)
