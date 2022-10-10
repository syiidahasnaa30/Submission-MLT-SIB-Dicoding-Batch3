# Laporan Proyek Machine Learning - Rosyiidah Hasnaa

## Domain Proyek
Jantung merupakan organ paling vital dalam tubuh manusia. Sekalipun manusia beristirahat, jantung tetap bekerja. Jantung bekerja untuk mengalirkan darah ke seluruh tubuh. Menurut *World Health Organization*, sepertiga dari penyebab kematian didunia dikarenakan adanya kegagalan pada fungsi jantung. Oleh karena itu, untuk membantu para penderita gagal jantung dalam memprediksi lebih awal adanya faktor faktor yang dapat menimbulkan gagal jantung, dibuatlah sebuah model machine learning. Dengan adanya model machine learning tersebut, akan membantu para penderita beberapa faktor seperti diabetes, hipertensi, hiperlipidemia untuk antisipasi lebih awal.[[1](https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/45652)]

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah disebutkan, berikut rumusan masalah yang akan diselesaikan pada proyek ini :
-   Bagaimana cara melakukan pra-pemrosesan data lahan sehingga dapat digunakan untuk membuat model yang baik?
-   Bagaimana cara membangun model _machine learning_ untuk mengklasifikasikan data lahan ke dalam jenis tanaman yang cocok ditanam pada lahan tersebut?

### Goals
Adapun tujuan dari proyek ini adalah sebagai berikut :
- Melakukan pra-pemrosesan data lahan agar dapat digunakan dalam membangun model.
- Melatih model machine learning untuk melakukan klasifikasi penyakit gagal jantung dengan akurasi >0,8 
    ### Solution statements
    Untuk mencapai tujuan dari proyek ini dilakukan beberapa proses berikut :
    - **Melakukan Pra-pemrosesan data**
    Pada proses pra-pemrosesan data dilakukan tiga tahap berikut, yaitu :
        - Melakukan encoding pada fitur kategorikal
        - Melakukan pembagian antara data training dan data testing
        - Melakukan scaling pada fitur numerik pada data training
    - **Membangun model machine learning**
    Pada proses ini, dilakukan uji coba untuk membangun beberapa model machine learning untuk diambil model dengan akurasi terbaik. Model machine learning tersebut diantaranya :
        - Logistic Regression
        - Naive Bayes Classifier
        - Decision Tree
        - Random Forest
        - XGBoost

## Data Understanding
Data yang digunakan pada proyek ini merupakan dataset online _Heart Failure Prediction_ yang tersedia pada kaggle dataset.[[2](https://www.kaggle.com/fedesoriano/heart-failure-prediction)] 
  
### Variabel-variabel pada heart failure dataset
Pada dataset yang telah diunduh, terdapat 918 baris dan 12 kolom. 12 kolom tersebut terdiri dari satu target kelas dan 11 fitur lainnya. Sebelas fitur lainnya terdiri dari  tujuh buah fitur numerik dan empat buah fitur kategorikal. Penjelasan lebih detail terkait setiap fiturnya sebagai berikut :  

- **Age** merupakan fitur numerik yang mendeskripsikan terkait umur pasien.
- **Sex** merupakan fitur kategorikal yang menjelaskan jenis kelamin pasien. Terdapat 2 kategori pada fitur ini yaitu M(Male) untuk laki-laki, dan F(Female) untuk perempuan
- **ChestPainType** merupakan fitur kategorikal yang menjelaskan tipe nyeri dada yang dialami pasien. Fitur ini terdiri dari 4 kategori yaitu TA(Typical Angina), ATA(Atypical Angina), NAP(Non-Anginal Pain dan ASY(Asymptomatic)
- **RestingBP** merupakan fitur numerik yang menyatakan nilai tekanan darah pasien saat beristirahat (mmHg)
- **Cholesterol** merupakan fitur numerik yang menyatakan tingkat kolesterol yang dialami pasien (mm/dl)
- **FastingBS** merupakan fitur numerik yang menyatakan tingkat gula darah pasien. Fitur ini hanya terdiri dari dua nilai yaitu 1 ketika gula dara melebihi 120 mg/dl dan 0 ketika gula darah kurang dari 120 mg/dl
- **RestingECG** merupakan fitur kategorikal yang menunjukkan hasil elektrodiagram ketika pasien beristirahat. Fitur ini terdiri dari 3 kategori yaitu N ketika pasien normal, ST ketika pasien memiliki kelainan gelombang ST-T dan LVH menunjukkan kemungkinan pasien mengalami hipertrofi ventrikel kiri
- **MaxHR** merupakan fitur numerik yang menyatakan nilai rata-rata maksimum detak jantung pasien bernilai antara 60 dan 202)
- **Exercise Angina** merupakan fitur kategorikal yang menyatakan induksi latihan angina. Fitur ini terdiri dari dua kategori yaitu Y jika iya, N jika tidak.
- **Oldpeak** merupakan fitur numerik yang menyatakan nilai numerik yang diukur dalam depresi
- **ST_Slope** merupakan fitur kategorikal yang menyatakan kemiringan ST latihan puncak. Fitur ini memiliki 3 kategori yaitu Up,Flat, dan down

Itulah sebelas fitur yang terdapat pada heart failure dataset, sedangkan target kelas dari dataset ini adalah kolom **HeartDisease**. Dimana kolom ini hanya memiliki dua nilai yaitu 1 apabila pasien mengidap gagal jantung dan 0 apabila pasien tidak mengidap gagal jantung.

Selain itu, pada proyek ini penulis juga melakukan tahap explorasi. Tahap explorasi data dilakukan dalam dua tahap, yaitu tahap _Univariate Analysis_ dan tahap _Mutivariate Analysis_.

**1. _Univariate Analysis_**

Pada tahap _univariate analysis_ proses yang dilakukan adalah mengekplorasi data untuk mengetahui persebaran setiap fiturnya dalam dataset yang tersedia.

![univariate_sex](https://user-images.githubusercontent.com/67575741/194702259-7171e2f4-c633-4091-84ac-bfab60185b58.png)

**Gambar 1. Persebaran fitur _sex_ pada dataset**

Gambar 1 merupakan diagram persebaran fitur sex dari dataset. Dari diagram tersebut dapat diketahui bahwa persebaran jumlah pasien dengan jenis kelamin laki-laki pada dataset mencapai 79% dan presentase pasien perempuan pada dataset mencapai 21%.

![univariate_chestpaintype](https://user-images.githubusercontent.com/67575741/194702345-a4008ac3-1a66-4a8b-bf15-3f79b61dd113.png)

**Gambar 2. Persebaran Fitur _Chest Pain Type_**

Dari data diatas, dapat diketahui bahwasanya persebaran tipe nyeri dada ASY menempati posisi terbesar dengan presentasi 54%. Kemudian, disusul dengan NAP sebesar 22,1%, tipe ATA sebesar 18.8% dan yang terakhir TA dengan presentase 5%.

![ecg univariate](https://user-images.githubusercontent.com/67575741/194702525-d1b84ee2-5435-4d4a-b97f-21b20bb21832.png)

**Gambar 3. Persebaran Fitur _Resting ECG_**

Pada diagram yang ditunjukkan oleh gambar 3 dapat diketahui bahwa persebaran hasil elektrodiagram pasien ketika beristirahat(resting ecg) normal sebesar 60,1%, sedangkan pasien yang mengalami kelaian gelombang ST sebesar 19,4% dan yang meiliki kemungkinan mengalami hipertrofi ventrikel kiri(LVH) sebesar 20,5%.

![ST Slope univariate](https://user-images.githubusercontent.com/67575741/194702542-b6abcd8b-6fef-416c-ae7d-ae4db2b04d6e.png)

**Gambar 4. Persebaran Fitur _ST Slope_**

Pada diagram yang ditunjukkan oleh gambar 4, dapat diketahui bahwa persebaran ST Slope flat sebesar 50,1%, Up sebesar 43%, dan down sebesar 6,9%.

![numerik univariate](https://user-images.githubusercontent.com/67575741/194702598-810669b0-65de-4f69-b6ee-f7d2864652f4.png)

**Gambar 5. Persebaran Fitur Numerik Pada Dataset**

Fitur numerik merupakan fitur yang bertipe numerik, fitur-fitur tersebut pada dataset meliputi fitur Age,RestingBP,Cholesterol,FastingBS,MaxHR,Oldpeak dan targetnya yang berupa nilai numerik yaitu HeartDisease. Pada diagram yang ditunjukkan oleh gambar 5 dapat diketahui bahwasanya dalam dataset pasien memiliki rentang usia antara 28-77 tahun, dan didominasi oleh rentang usia 50-60 tahun. Sedangkan untuk resting bp pasien rata-rata didominasi pada rentang 100-175. Sedangkan tingkat kolesterol pasien pada dataset didominasi pada rentang 100-400 an. Sedangkan yang memiliki tingkat kolesterol sebesar nol mencapai 170-an pasien. Pada data detak jantung, maksimal detak jantung pasien berada pada rentang 100-200. Sedangkan untuk oldpeak berada pada rentang 0-4. Sedangkan untuk fasting bs yang memiliki nilai lebih dari 120 mmHg mencapai 200 pasien, selebihnya fasting bs tidak mencapai 10 mmHg. Pada grafik terakhir, dapat diketahui persebaran pasien yang memiliki penyakit gagal jantung mencapai 500 pasien, sedangkan yang tidak mengidap penyakit gagal jantung mencapai 400 pasien.

**2. _Mutivariate Analysis_**

Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Berfungsi untuk mengetahui hubungan antar fitur pada dataset.

![sex_multi](https://user-images.githubusercontent.com/67575741/194702620-bedf1ef7-8474-4b89-a838-8aa43006f904.png)
![chestpaintype multi](https://user-images.githubusercontent.com/67575741/194702623-71063aae-eabb-428d-983b-9931aa6c4c75.png)
![resting ecg multi](https://user-images.githubusercontent.com/67575741/194702631-98cc218b-0b80-407c-8c1b-7a8bc5cf8616.png)
![exercise angine multi](https://user-images.githubusercontent.com/67575741/194702639-9947d249-1f0d-4ff7-b0c4-5b123a5c3522.png)
![st slope mlti](https://user-images.githubusercontent.com/67575741/194702655-dc7c194b-279d-434a-8fce-05bf85b7fac5.png)

**Gambar 6. Keterkaitan Setiap Fitur Kategorikal dengan HearDisease**

Dari ilustrasi yang ditunjukkan oleh gambar 6, dapat diketahu bahwasanya sebagian besar penderita gagal jantung adalah pasien laki-laki.Apabila berdasarkan jenis nyeri jantung yang dialami kebanyakn pengidap gagal jantung memiliki jenis nyeri dada ASY. Sedangkan untuk keterkaitan antara RestingECG dengan penyakit gagal jantung cenderung tidak adanya keterkaitan, karena persebarannya hampir rata. Untuk exercise angina cenderung berpengaruh pada penyakit gagal jantung. Sedangkan untuk ST Slope penderita gagal jantung didominasi oleh pasien yang memiliki ST Slope normal.

Untuk persebaran pasien yang mengidap gagal jantung berdasarkan fitur numerik seperti yang tampak pada gambar dibawah

![numerik multi](https://user-images.githubusercontent.com/67575741/194702665-f3b3845e-c6aa-4726-b7a2-ab3372ed5766.png)

**Gambar 7. Keterkaitan setiap fitur numerik dengan HeartDisease**

Selain itu, untuk melihat korelasi antar fitur pada proyek ini penulis juga memanfaatkan matriks kolerasi yangg didapatkan melalui fungsi corr yang tersedia pada library pandas. Pada matriks korelasi, warna gelap menandakan korelasi negatif, sedangkan warna terang menandakan korelasi positif. 

![newplot](https://user-images.githubusercontent.com/67575741/194702678-ffe7daa9-7b67-40bc-8a07-c6486cd94766.png)

**Gambar 8. Matriks korelasi antar fitur numerik**

Pada gambar 8, matriks korelasi menunjukkan bahwa heart disease memiliki korelasi negatif dengan Cholesterol dan MaxHR. Dan Heart disease memiliki korelasi positif dengan fitur Oldpeak,FastingBS dan Age. 

## Data Preparation
Berikut ini adalah tahapan pra-premosesan data yang dilakukan pada proyek ini:
- **Melakukan encoding pada fitur kategorikal**
   Untuk dapat diproses dalam sebuah model, terlebih dahulu seluruh fitur yang akan masuk diubah dahulu kedalam bentuk numerik. Maka dari itu, diperlukanlah sebuah proses encdoing untuk mengubah fitur fitur yang bertipe kategorikal menjadi numerikal. Untuk itu sebelum melakukan proses encding, terlebih dahulu kita pisahkan fitur-fitur kategorikal dan fitur-fitur numerikal. Untuk melakukan proses encoding, pada proyek ini penulis memanfaatkan One Hot Encoder yang telah tersedia pada library scikit learning.
  
- **Melakukan pembagian dataset**
    Sebelum melangkan pada proses pemodelan, terlebih dahulu kita harus membedakan data training dan data testing. Hal ini dilakukan untuk mempertahankan sebagian data yang ada untuk menguji seberapa baik model yang kita buat terhadap data baru. Pada proses ini data testing berperan sebagai data baru, sehingga proses lainnya pada data latih tidak perlu dilakukan pada data testing.Pada proyek ini, penulis memanfaatkan fungsi train test split dari library sklearn untuk melakukan pembagian data trainng dan data testing. Dimana ukuran dari data testing sebesar 20% dari ukuran dataset. Sehingga dari 918 dari jumlah total sample dataset, sebesar 734 data berperan sebagai data training, dan 184 data berperan sebagai data testing.

- **Standardisasi data pada semua fitur numerik pada dataset**
  Model machine learning akan memiliki performa yang baik apabila dimodelkan dengan data dengan skala yang relatif sama atau memiliki distribusi data yang normal. Untuk itu, diperlukan proses scaling dan standarisasi agar data dapat berubah menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, penulis menggunakan StandardScaler untuk melakukan proses standarisasi. Proses stanrisasi hanya dilakukan pada fitur numerik. Untuk proses standarisasi sendiri melakukan proses standarisasi dengan mengurangi nilai asal dengan nilai rata-rata suatu fitur, lalu membanginya dengan standar deviasi untuk menggeser distribusi. Rumus dibawah ini merupakan rumus dari standart scaler.
  
  $$ X scaled = { x - mean \over std } $$

## Modeling
Pada proyek ini, model yang dibuat merupakan kasus binary classification. Untuk mendapatkan model dengan performa terbaik, penulis melakukan percobaan pada beberpada model klasifikasi. Model klasifikasi tersebut diantaranya :
- **Logistic Regression**
Logistic regression merupakan salah satu algoritma regresi yang dapat dimanfaatkan untuk proses klasifikasi. Algoritma ini berfungsi untuk mencari hubungan antar fitur(input) diskrit/kontinu dengan probabiltas hasil output diskrit tertentu. Berdasarkan tipenya, logistic regression dibagi menjadi tiga tipe[[3](https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603)] :
    - Binary Logistic Regression
    Jenis logistic regression yang hanya memiliki dua output
    - Multinomial Logistic Regression
    Logistic regression yang memiliki 2 output atau lebih tapi tidak memperhatikan urutan.
    - Ordinal Logistic Regression
    Jenis logistic regression yang memiliki dua output atau lebih dengan memperhatikan urutan.

Proses pembangunan model dari logistic regression memanfaatkan logistic function. Logistic function adalah suatu fungsi yang dibentuk dengan menyamakan nilai Y pada linear function dengan nilai Y pada sigmoid function. Tujuan dari logistic function untuk merepresentasikan data yang dimiliki kedalam bentuk sigmoid function. Langkah -langkah dalam membentuk logisctic function adalah sebagai berikut :
    - Melakukan opersai Invers pada Sigmoid Function, sehingga fungsi sigmoid berubah bentuk menjadi Y = ln(p/(1-p).
    - Setarakan dengan fungsi Linear Y = b0+b1*X sehingga kita dapatkan persamaan ln(p/(1-p) = b0+b1*X.
    - Ubah persamaan ln(p/(1-p) = b0+b1*X kedalam bentuk logaritmik sehingga didapatkan persamaan P = 1/(1+e^-(b0+b1*X))
 
 Pada proyek ini untuk melakukan pemodelan cuku memanggil LogisticRegression dari library sklearn. Kemudian, parameter yang dimasukkan kedalam model adalah data training dan juga label dari data training. Hasil evaluasi dari pemodelan logistic regression pada proyek ini adalah sebagai berikut :
| Nama Algoritma | Accuracy | Precision | Recall | F1-Score
| ------ | ------ |------ |------ |------ |
| Logistic Regression| 0.53	| 0.94 | 0.17 | 0.28 |

- **Naive Bayes Classifier**
Naive bayes classifier merupakan salah satu metode klasifikasi yang berdasarkan pada teorema bayes. Teorema bayes merupakan sebuah teori untuk memprediksi peluang dimasa mendatang berdasarkan pengalaman di masa sebelumnya.[[4](https://binus.ac.id/bandung/2019/12/algoritma-naive-bayes/)] Kelebihan dan kekurangan dari metode ini adalah sebagai berikut :
    - Kelebihan :
        - Bisa dipakai untuk data kuantitatif maupun kualitatif
        - Tidak memerlukan jumlah data yang banyak
        - Tidak perlu melakukan data training yang banyak
        - Jika ada nilai yang hilang, maka bisa diabaikan dalam perhitungan.
        - Perhitungannya cepat dan efisien
        - Bisa digunakan untuk klasifikasi masalah biner ataupun multiclass
    - Kekurangan :
        - Apabila probabilitas kondisionalnya bernilai nol, maka probabilitas prediksi juga akan bernilai nol
        - Asumsi bahwa masing-masing variabel independen membuat berkurangnya akurasi, karena biasanya ada korelasi antara variabel yang satu dengan variabel yang lain
        - Keakuratannya tidak bisa diukur menggunakan satu probabilitas saja. Butuh bukti-bukti lain untuk membuktikannya.
        - Untuk membuat keputusan, diperlukan pengetahuan awal atau pengetahuan mengenai masa sebelumnya. Keberhasilannya sangat bergantung pada pengetahuan awal tersebut Banyak celah yang bisa mengurangi efektivitasnya
        - Dirancang untuk mendeteksi kata-kata saja, tidak bisa berupa gambar
- **Decision Tree**
Decision tree merupakan salah satu algoritma supervised learning untuk memecahkan masalah klasifikasi. Decision tree melakukan prediksi suatu kelas berdasarkan aturan-aturan yang dibentuk setelah mempelajari data yang ada. Konsep dari proses decision tree adalah mengkonversi sebuah data menjadi sebuah pohon keputusan dan ketetapan.[[5](https://medium.com/@raihanaglest/pemahaman-decision-tree-3cb3ab1a27c9)]
    - Kelebihan dari algoritma decision tree adalah sebagai berikut :
        - Dapat mencegah timbulnya sebuah masalah dalam analisis multivariant tanpa mengurangi kualitas keputusan yang dihasilkan.
        - Perubahaan pada distrik pengambilan keputusan yang bersifat kompleks dan global menjadi lebih mudah dan jelas.
        - Adanya sample uji yang hanya berlandaskan pada kriteria atau kelas tertentu.
        - Lebih mudah untuk memperoleh features dari node dalam yang berbeda, karena feature ini akan membedakan kriteria kemudian membandingkan kriteria yang lain di dalam node yang sama.
    - Kekurangan algoritma decision tree :
        - Terjadinya overlap karena variabel yang digunakan begitu banyak.
        - Kesulitan dalam memodelkan pohon keputusan.
        -  Besarnya pengumpulan jumlah error dari setiap level.

- **Random Forest**
Random Forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar.Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada kedalam decision tree secara acak. Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik [[6](https://id.wikipedia.org/wiki/Random_forest)]. 
    - Kelebihan algoritma Random Forest adalah sebagai berikut :
        -   Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat
        -   Berjalan secara efisien pada data besar
        -   Dapat menangani ribuan variabel input tanpa penghapusan variabel
        -   Memberikan perkiraan variabel apa yang penting dalam klasifikasi
        -   Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang
    -   Kekurangan dari algoritma Random Forest adalah sebagai berikut :
        -   Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang bising/noise
        -   Untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.

- **XgBoost**
XGboost adalah algoritma yang merupakan implementasi lanjutan dari algoritma peningkatan gradien (Gradient Boosting). XGboost menggunakan prinsip ensemble yaitu menggabungkan beberapa set pembelajar (tree) yang lemah menjadi sebuah model yang kuat sehinga menghasilkan prediksi yang kuat. [[7](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)]) :
    -   Kelebihan Algoritma XGboost sebagai berikut :
        -   Dapat melakukan pemrosesan paralel yang dapat mempercepat komputasi
        -   Memiliki fitur regularisasi untuk mencegah overfitting
        -   Menangani berbagai jenis pola sparsity dalam data dengan lebih efisien
        -   Dilengkapi dengan built in cross validation

Pada semua algoritma yang dilakukan, data training dimasukkan pada model secara langsung setelah melewati tahap pre-processing. Namun, pada algoritma decision tree terdapat parameter tambahan berupa criterion yang diisi dengan entropy. Sedangka pada random forest terdapat parameter n_estimator sebesar 200 dan criterion juga entropi. Dari kelima algoritma tersebut, didapatkan hasil metric akurasi sebagai berikut :

**Tabel 1. Tabel Metrik Evaluasi Model**
| Nama Algoritma | Accuracy | Precision | Recall | F1-Score
| ------ | ------ |------ |------ |------ |
| Logistic Regression| 0.53	| 0.94 | 0.17 | 0.28 |
| Naive Bayes | 0.56	| 0.56 | 1.0 | 0.72 |
| Decision Tree | 0.82	| 0.86 | 0.81 | 0.83 |
| Random Forest | 0.86	| 0.84 | 0.92 | 0.88 |
| XgBoost | 0.82	| 0.85 | 0.82 | 0.83 |

Berdasarkan hasil yang tertera pada tabel 1, didapatkan bahwasanya nilai akurasi tertinggi diraih oleh algoritma random forest sebesar 0.86 atau setara dengan 86% dan disusul oleh algoritma XGBoost dengan akurasi sebesar 0,82 setara dengan 82%. Dari pelatihan model tersebut didapatkan bahwasanya algoritma RandomForest dan XgBoost mampu digunakan untuk melakukan klasifikasi penyakit gagal jantung dan memiliki akurasi melebihi 80%.

## Evaluation
Proyek ini mengangkat tema mengenai klasifikasi, sehingga pada proyek ini metrik evaluasi yang digunakan merupakan metrik evaluasi yang umumnya digunakan pada masalah klasifikasi pada machine learning. Metrik evaluasi tersebut meliputi accuracy, precision, recall, dan f-1 score. Keempat metric tersebut telah tersedia pada library scikitlearn dengan memanggil sklearn.metrics import accuracy_score, precision_score, recall_score dan f1_score. Untuk memanfaatkan keempat metrik tersebut dibutuhkan parameter berupa label dari data validasi dan label dari hasil prediksi data validasi. Penjelasan dari masing -masing metrik tersebut sebagai berikut :

- **Accuracy**

Accuracy merupakan sebuah metrik yang menunjukkan presentase seberapa sering model yang telah kita bangun menunjukkan hasil prediksi yang sesuai dengan target dari keseluruhan data. Baik itu prediksi positif maupun prediksi negatif, namun keduanya mampu diprediksi dengan benar. Apabila dihitung menggunakan rumus matematika, rumusan dari accuracy adalah sebagai berikut :

$$ Accuracy = { TP + TN \over TP + TN + FP + FN} $$

**TN :** Terjadi ketika model memprediksi hasil negatif, dan pada kenyatannya target class label juga negatif. Pada proyek ini, TN terjadi apabila model memprediksi seorang pasien tidak mengidap gagal ginjal dan pada kenyataannya pasien memang tidak mengidap gagal ginjal.

**TP :** Terjadi ketika model memprediksi hasil positif, dan pada kenyatannya target class label juga positif. Pada proyek ini, TP terjadi apabila model memprediksi seorang pasien mengidap gagal ginjal dan pada kenyataannya pasien memang mengidap gagal ginjal.

**FN :** Terjadi ketika model memprediksi hasil negatif, dan pada kenyatannya target class label bernilai positif. Pada proyek ini, FN terjadi apabila model memprediksi seorang pasien tidak mengidap gagal ginjal dan pada kenyataannya pasien mengidap gagal ginjal.

**FP :** Terjadi ketika model memprediksi hasil positif, dan pada kenyatannya target class label bernilai negatif. Pada proyek ini, FP terjadi apabila model memprediksi seorang pasien mengidap gagal ginjal dan pada kenyataannya pasien tidak mengidap gagal ginjal.

Berdasarkan rumusan tersebut, masing -masing algoritma yang telah dibangun pada proyek ini memiliki nilai accuracy sebagai berikut :

**Tabel 2. Hasil Evaluasi Model Menggunakan Metrik Accuracy**
| Nama Algoritma | Accuracy | 
| ------ | ------ |
| Logistic Regression| 0.53	|
| Naive Bayes | 0.56	|
| Decision Tree | 0.82	|
| Random Forest | 0.86	|
| XgBoost | 0.82	|

Dari data yang ditunjukkan  oleh tabel 2, dapat kita simpulkan bahwasanya nilai accuracy tertinggi diraih oleh algoritma random forest yang kemudian disusul oleh decision tree dan xgboost. Ketiga algoritma tersebut telah memenuhi tujuan dari proyek ini yaitu mampu memprediksi dengan accuracy melebihi 0,8.

- **Precision**

Precision dapat didefinisikan sebagai presentase dari hasil prediksi positif yang diprediksi dengan benar dari keseluruhan jumlah prediksi positif baik yang diprediksi dengan benar ataupun salah. Apabila dirumuskan dalam bentuk matematika rumusan dari precision adalah sebagai berikut:

$$ Precision = { TP \over TP + FP } $$

Berdasarkan rumusan tersebut, masing -masing algoritma yang telah dibangun pada proyek ini memiliki nilai precision sebagai berikut :

**Tabel 3. Tabel Evaluasi Model Menggunakan Metrik Precision**

| Nama Algoritma | Precision | 
| ------ | ------ |
| Logistic Regression| 0.94 |
| Naive Bayes | 0.56 |
| Decision Tree | 0.86 | 
| Random Forest | 0.84 |
| XgBoost |0.85 | 0.82 |

Dari data yang ditunjukkan  oleh tabel 3, dapat kita simpulkan bahwasanya nilai precision tertinggi diraih oleh algoritma logistic regression yang kemudian disusul oleh decision tree. Umumnya dari algoritma logistic regression, decision tree, random forest, juga xgboost memiliki precision diatas 0,8. Hanyasatu algoritma saja yaitu naive bayes yang memiliki precision dibawah 0,8.

- **Recall**

Recall dapat didefinisikan sebagai presentase positif yang diprediksi dengan benar dari semua hasil positif yang sebenarnya.Recall disebut juga denga sensitivitas. Rumusan dari recall adalah sebagai berikut:

$$ Recall = { TP \over TP + FN } $$

Berdasarkan rumusan tersebut, masing -masing algoritma yang telah dibangun pada proyek ini memiliki nilai recall sebagai berikut :

**Tabel 4. Tabel Evaluasi Model Menggunakan Metrik Recall**

| Nama Algoritma |  Recall |
| ------ | ------|
| Logistic Regression| 0.17 |
| Naive Bayes | 1.0 |
| Decision Tree | 0.81 |
| Random Forest | 0.92 |
| XgBoost | 0.82 |

Dari data yang ditunjukkan  oleh tabel 4, dapat kita simpulkan bahwasanya nilai recall tertinggi diraih oleh algoritma naive bayes. Dan nilai recall terendah dimiliki oleh logistic regression.

- **F1-Score**

F1-Score didapatkan dari hasil perkalian antara precision dan recall yang kemudian dibagi dengan penjumlahan antara precision dengan recall lalu dikalikan dua. F-1 Score cocok digunakan pada dataset yang persebaran antara label positif dan negatif tidak merata.Rumusan dari f-1 score adalah sebagai berikut:

$$ F-1 Score = { 2 * (Recall * Precision) \over (Recall + Precision) } $$

Berdasarkan rumusan tersebut, masing -masing algoritma yang telah dibangun pada proyek ini memiliki nilai f-1 score sebagai berikut :

**Tabel 5. Tabel Evaluasi Model Menggunakan Metrik F-1 Score**
| Nama Algoritma |F1-Score
| ------ | ------ |
| Logistic Regression|  0.28 |
| Naive Bayes | 0.72 |
| Decision Tree | 0.83 |
| Random Forest | 0.88 |
| XgBoost | 0.83 |

Dari data yang ditunjukkan  oleh tabel 5, dapat kita simpulkan bahwasanya nilai f-1 score tertinggi diraih oleh algoritma random forest dan disusul oleh decision tree dan xgboost. Sedangkan untuk nilai f-1 score terendah ada pada algoritma logistic regression.

## Referensi
[1] Adi, S., & Wintarti, A. (2022). KOMPARASI METODE SUPPORT VECTOR MACHINE (SVM), K-NEAREST NEIGHBORS (KNN), DAN RANDOM FOREST (RF) UNTUK PREDIKSI PENYAKIT GAGAL JANTUNG. MATHunesa: Jurnal Ilmiah Matematika, 10(2), 258-268.

[2] fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved September 27, 2022,from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

[3] Michael, V. (2019, May 9). Machine Learning: Mengenal Logistic Regression. Retrieved October 8, 2022, from Medium: https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603

[[4](https://binus.ac.id/bandung/2019/12/algoritma-naive-bayes/)] Algoritma Naive Bayes

[[5](https://medium.com/@raihanaglest/pemahaman-decision-tree-3cb3ab1a27c9)] Pemahaman Decision Tree

[[6](https://id.wikipedia.org/wiki/Random_forest)] Random Forest

[[7](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)] XGBoost Algorithm: Long May She Reign




