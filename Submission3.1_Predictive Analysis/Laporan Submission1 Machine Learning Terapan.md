# Laporan Proyek Machine Learning - Rosyiidah Hasnaa

## Domain Proyek

Jantung merupakan organ paling vital dalam tubuh manusia. Sekalipun manusia beristirahat, jantung tetap bekerja. Jantung bekerja untuk mengalirkan darah ke seluruh tubuh. Menurut *World Health Organization*, sepertiga dari penyebab kematian didunia dikarenakan adanya kegagalan pada fungsi jantung. Oleh karena itu, untuk membantu para penderita gagal jantung dalam memprediksi lebih awal adanya faktor faktor yang dapat menimbulkan gagal jantung, dibuatlah sebuah model machine learning. Dengan adanya model machine learning tersebut, akan membantu para penderita beberapa faktor seperti diabetes, hipertensi, hiperlipidemia untuk antisipasi lebih awal.[Komparasi Metode Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Dan Random Forest (RF) Untuk Prediksi Penyakit Gagal Jantung](https://ejournal.unesa.ac.id/index.php/mathunesa/article/view/45652)


## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah disebutkan, berikut rumusan masalah yang akan diselesaikan pada proyek ini :
- Bagaimana cara membuat prediksi penyakit gagal jantung pada manusia dengan menggunakan _machine learning_?
- Berapa nilai akurasi terbaik yang didapat dengan menggunakan _machine learning_?

### Goals

Adapun tujuan dari proyek ini adalah sebagai berikut :
- Mengetahui model machine learning yang dapat digunakan untuk memprediksi penyakit gagal jantung
- Mendapatkan model machine learning dengan akurasi terbaik


**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Data yang digunakan pada proyek ini merupakan dataset online _Heart Failure Prediction_ Kaggle yang telah tersedia pada [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction).
  
Pada dataset yang telah diunduh, terdapat 918 baris dan 12 kolom. 12 kolom tersebut terdiri dari satu target kelas dan 11 fitur lainnya. Sebelas fitur lainnya terdiri dari  tujuh buah fitur numerik dan empat buah fitur kategorikal. Penjelasan lebih detail terkait setiap fiturnya sebagai berikut :  

- **Age** merupakan fitur numerik yang mendeskripsikan terkait umur pasien.
- **Sex** merupakan fitur kategorikal yang menjelaskan jenis kelamin pasien. Terdapat 2 kategori pada fitur ini yaitu M(Male) untuk laki-laki, dan F(Female) untuk perempuan
- **ChestPainType** merupakan fitur kategorikal yang menjelaskan tipe nyeri dada yang dialami pasien. Fitur ini terdiri dari 4 kategori yaitu TA(Typical Angina), ATA(Atypical Angina), NAP(Non-Anginal Pain dan ASY(Asymptomatic)
- **RestingBP** merupakan fitur numerik yang menyatakan nilai tekanan darah pasien saat beristirahat (mmHg)
- **Cholesterol** merupakan fitur numerik yang menyatakan tingkat kolesterol yang dialami pasien (mm/dl)
- **FastingBS** merupakan fitur numerik yang menyatakan tingkat gula darah pasien. Fitur ini hanya terdiri dari dua nilai yaitu 1 ketika gula darah pasien melebihi 120 mg/dl dan 0 ketika gula darah pasien kurang dari 120 mg/dl
- **RestingECG** merupakan fitur kategorikal yang menunjukkan hasil elektrodiagram ketika pasien beristirahat. Fitur ini terdiri dari 3 kategori yaitu N ketika pasien normal, ST ketika pasien memiliki kelainan gelombang ST-T dan LVH menunjukkan kemungkinan pasien mengalami hipertrofi ventrikel kiri
- **MaxHR** merupakan fitur numerik yang menyatakan nilai rata-rata maksimum detak jantung pasien bernilai antara 60 dan 202)
- **Exercise Angina** merupakan fitur kategorikal yang menyatakan induksi latihan angina. Fitur ini memiliki dua kategori Y jika iya, N jika tidak.
- **Oldpeak** merupakan fitur numerik yang menyatakan nilai numerik yang diukur dalam depresi.
- **ST_Slope** merupakan fitur kategorikal yang menyatakan kemiringan ST latihan puncak. Fitur ini memiliki 3 kategori Up,Flat, dan down

Itulah sebelas fitur yang terdapat pada heart failure dataset, sedangkan target
HeartDisease : merupakan target kesimpulan apakah pasien mengidap gagal jantung atau tidak [1: apabila pasien mengidap gagal jantung, 0: apabila pasien tidak mengidap gagal jantung]

Pada proyek ini, penulis juga melakukan exploratory data analysis untuk mengetahui persebaran setiap fitur dalam dataset. Pada tahap ini, terdapat dua jenis tahapan yang dilakukan yaitu _univariate analysis_ dan _multivariate analysis_.

**1. _Univariate Analysis_**
Pada tahap ini dilakukan explorasi untuk mengetahui persebaran setiap fitu dalam dataset.

* Persebaran Fitur _Sex_ pada dataset
![Image of Dataset](url)

Dari data diatas dapat diketahui bahwa persebaran jumlah pasien dengan jenis kelamin laki-laki pada dataset mencapai 79% dan presentase pasien perempuan pada dataset mencapai 21%

* Persebaran Fitur _Chest Pain Type_
![Image of Dataset](url)

Darii data diatas, dapat diketahui bahwasanya persebaran tipe ASY menempati posisi terbesar dengan presentasi 54%. Kemudian, disusul dengan NAP sebesar 22,1%, tipe ATA sebesar 18.8% dan yang terakhir TA dengan presentase 5%.

* Persebaran Fitur _Resting ECG_
![Image of Dataset](url)

Dari data diatas, dapat diketahui bahwa persebran resting ecg normal sebesar 60,1%, ST 19,4% dan LVH 20,5%

* Persebaran Fitur _ST Slope_
![Image of Dataset](url)

Dari data diatas, dapat diketahui bahwa persebaran ST Slope flat sebesar 50,1%, Up sebesar 43%, dan down sebesar 6,9%.

* Persebaran Fitur Numerik
Fitur numerikal meliputi Age,RestingBP,Cholesterol,FastingBS,MaxHR,Oldpeak dan targetnya yang berupa nilai numerik yaitu HeartDisease

![Image of Dataset](url)

Dari data tersebut dapat diketahui bahwasanya dalam dataset pasien memiliki rentang usia antara 28-77 tahun, dan didominasi oleh rentang usia 50-60 tahun. Sedangkan untuk resting bp pasien rata-rata didominasi pada rentang 100-175. Sedangkan tingkat kolesterol pasien pada dataset didominasi pada rentang 100-400 an. Sedangkan yang memiliki tingkat kolesterol sebesar nol mencapai 170-an pasien. Pada data detak jantung, maksimal detak jantung pasien berada pada rentang 100-200. Sedangkan untuk oldpeak berada pada rentang 0-4. Sedangkan untuk fasting bs yang memiliki nilai lebih dari 120 mmHg mencapai 200 pasien, selebihnya fasting bs tidak mencapai 10 mmHg. Pada grafik terakhir, dapat diketahui persebaran pasien yang memiliki penyakit gagal jantung mencapai 500 pasien, sedangkan yang tidak mengidap penyakit gagal jantung mencapai 400 pasien.

**2. _Mutivariate Analysis_**
Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Berfungsi untuk mengetahui hubungan antar fitur pada dataset.

[Image Dataset]()
Dari ilustrasi diatas, dapat diketahu bahwasanya sebagian besar penderita gagal jantung adalah pasien laki-laki.Apabila berdasarkan jenis nyeri jantung yang dihapai kebanyakn pengidap gagal jantung memiliki jenis nyeri dada ASY. Sedangkan untuk keterkaitan antara RestingECG dengan penyakit gagal jantung cenderung tidak adanya keterkaitan, karena persebarannya hampir rata. Untuk exercise angina cenderung berpengaruh pada penyakit gagal jantung. Sedangkan untuk ST Slope penderita gagal jantung didominasi oleh pasien yang memiliki ST Slope normal.


Untuk persebaran pasien yang mengidap gagal jantung berdasarkan fitur numerik seperti yang tampak pada gambar dibawah
! [Image dataset](url)


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
