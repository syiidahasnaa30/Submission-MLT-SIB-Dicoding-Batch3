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
* Persebaran Fitur _Sex_ pada dataset
![Image of Dataset](url)
Dari data diatas dapat diketahui bahwa persebaran jumlah pasien dengan jenis kelamin laki-laki pada dataset mencapai 79% dan presentase pasien perempuan pada dataset mencapai 21%.

* Persebaran Fitur _Chest Pain Type_ 
![Image of Dataset](url)
Dari data diatas, dapat diketahui bahwasanya persebaran tipe nyeri dada ASY menempati posisi terbesar dengan presentasi 54%. Kemudian, disusul dengan NAP sebesar 22,1%, tipe ATA sebesar 18.8% dan yang terakhir TA dengan presentase 5%.

* Persebaran Fitur _Resting ECG_
![Image of Dataset](url)
Dari data diatas, dapat diketahui bahwa persebaran hasil elektrodiagram pasien ketika beristirahat(resting ecg) normal sebesar 60,1%, sedangkan pasien yang mengalami kelaian gelombang ST sebesar 19,4% dan yang meiliki kemungkinan mengalami hipertrofi ventrikel kiri(LVH) sebesar 20,5%.

* Persebaran Fitur _ST Slope_
![Image of Dataset](url)
Dari data diatas, dapat diketahui bahwa persebaran ST Slope flat sebesar 50,1%, Up sebesar 43%, dan down sebesar 6,9%.

* Persebaran Fitur Numerik
Fitur numerikal meliputi Age,RestingBP,Cholesterol,FastingBS,MaxHR,Oldpeak dan targetnya yang berupa nilai numerik yaitu HeartDisease
![Image of Dataset](url)
 
 Dari data tersebut dapat diketahui bahwasanya dalam dataset pasien memiliki rentang usia antara 28-77 tahun, dan didominasi oleh rentang usia 50-60 tahun. Sedangkan untuk resting bp pasien rata-rata didominasi pada rentang 100-175. Sedangkan tingkat kolesterol pasien pada dataset didominasi pada rentang 100-400 an. Sedangkan yang memiliki tingkat kolesterol sebesar nol mencapai 170-an pasien. Pada data detak jantung, maksimal detak jantung pasien berada pada rentang 100-200. Sedangkan untuk oldpeak berada pada rentang 0-4. Sedangkan untuk fasting bs yang memiliki nilai lebih dari 120 mmHg mencapai 200 pasien, selebihnya fasting bs tidak mencapai 10 mmHg. Pada grafik terakhir, dapat diketahui persebaran pasien yang memiliki penyakit gagal jantung mencapai 500 pasien, sedangkan yang tidak mengidap penyakit gagal jantung mencapai 400 pasien.

**2. _Mutivariate Analysis_**
Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Berfungsi untuk mengetahui hubungan antar fitur pada dataset.

![Image Dataset](url)
Dari ilustrasi diatas, dapat diketahu bahwasanya sebagian besar penderita gagal jantung adalah pasien laki-laki.Apabila berdasarkan jenis nyeri jantung yang dialami kebanyakn pengidap gagal jantung memiliki jenis nyeri dada ASY. Sedangkan untuk keterkaitan antara RestingECG dengan penyakit gagal jantung cenderung tidak adanya keterkaitan, karena persebarannya hampir rata. Untuk exercise angina cenderung berpengaruh pada penyakit gagal jantung. Sedangkan untuk ST Slope penderita gagal jantung didominasi oleh pasien yang memiliki ST Slope normal.

Untuk persebaran pasien yang mengidap gagal jantung berdasarkan fitur numerik seperti yang tampak pada gambar dibawah
![Image dataset](url)

Selain itu, itu melihat korelasi antar fitur pada proyek ini penulis juga memanfaatkan matriks kolerasi yangg didapatkan melalui fungsi corr yang tersedia pada library pandas. Pada matriks korelasi, warna gelap menandakan korelasi negatif, sedangkan warna terang menandakan korelasi positif. 
![Image Dataset](url)
Hasil diatas menunjukkan bahwa heart disease memiliki korelasi negatif dengan Cholesterol dan MaxHR. Dan Heart disease memiliki korelasi positif dengan fitur Oldpeak,FastingBS dan Age. 

## Data Preparation
Berikut ini adalah tahapan pra-premosesan data yang dilakukan pada proyek ini:
- **Melakukan encoding pada fitur kategorikal**
   Untuk dapat diproses dalam sebuah model, terlebih dahulu seluruh fitur yang akan masuk diubah dahulu kedalam bentuk numerik. Maka dari itu, diperlukanlah sebuah proses encdoing untuk mengubah fitur fitur yang bertipe kategorikal menjadi numerikal. Untuk itu sebelum melakukan proses encding, terlebih dahulu kita pisahkan fitur-fitur kategorikal dan fitur-fitur numerikal. Untuk melakukan proses encoding, pada proyek ini penulis memanfaatkan One Hot Encoder yang telah tersedia pada library scikit learning.
  
- **Melakukan pembagian dataset**
    Sebelum melangkan pada proses pemodelan, terlebih dahulu kita harus membedakan data training dan data testing. Hal ini dilakukan untuk mempertahankan sebagian data yang ada untuk menguji seberapa baik model yang kita buat terhadap data baru. Pada proses ini data testing berperan sebagai data baru, sehingga proses lainnya pada data latih tidak perlu dilakukan pada data testing.Pada proyek ini, penulis memanfaatkan fungsi train test split dari library sklearn untuk melakukan pembagian data trainng dan data testing. Dimana ukuran dari data testing sebesar 20% dari ukuran dataset. Sehingga dari 918 dari jumlah total sample dataset, sebesar 734 data berperan sebagai data training, dan 184 data berperan sebagai data testing.

- **Standardisasi data pada semua fitur numerik pada dataset**
  Model machine learning akan memiliki performa yang baik apabila dimodelkan dengan data dengan skala yang relatif sama atau memiliki distribusi data yang normal. Untuk itu, diperlukan proses scaling dan standarisasi agar data dapat berubah menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, penulis menggunakan StandardScaler untuk melakukan proses standarisasi. Proses stanrisasi hanya dilakukan pada fitur numerik. Untuk proses standarisasi sendiri melakukan proses standarisasi dengan mengurangi nilai asal dengan nilai rata-rata suatu fitur, lalu membanginya dengan standar deviasi untuk menggeser distribusi. Gambar dibawah merupakan rumus dari standard scaler.![Standard Scaler Image](https://th.bing.com/th/id/OIP.aZ4K__oNEvn7VQWi7sb9iAHaB0?pid=ImgDet&rs=1)

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
![Tabel Matriks Akurasi]()

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

## Referensi
[[1](https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/45652)] Adi, S., & Wintarti, A. (2022). KOMPARASI METODE SUPPORT VECTOR MACHINE (SVM), K-NEAREST NEIGHBORS (KNN), DAN RANDOM FOREST (RF) UNTUK PREDIKSI PENYAKIT GAGAL JANTUNG. MATHunesa: Jurnal Ilmiah Matematika, 10(2), 258-268.
[[2](https://www.kaggle.com/fedesoriano/heart-failure-prediction)] fedesoriano. (September 2021). Heart Failure Prediction Dataset.
[[3](https://vincentmichael089.medium.com/machine-learning-2-logistic-regression-96b3d4e7b603)] Machine Learning: Mengenal Logistic Regression
[[4](https://binus.ac.id/bandung/2019/12/algoritma-naive-bayes/)] Algoritma Naive Bayes
[[5](https://medium.com/@raihanaglest/pemahaman-decision-tree-3cb3ab1a27c9)] Pemahaman Decision Tree
[[6](https://id.wikipedia.org/wiki/Random_forest)] Random Forest
[[7](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)] XGBoost Algorithm: Long May She Reign




