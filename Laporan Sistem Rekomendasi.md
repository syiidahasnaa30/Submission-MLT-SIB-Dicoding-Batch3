# Laporan Proyek Machine Learning - Rosyiidah Hasnaa

## Project Overview

Pandemi yang terjadi pada awal tahun 2020, menyebabkan perubahan besar dalam kehidupan masyarakat. Pemberhentian operasional bioskop ketika pandemi, menyebabkan peralihan perilaku masyarakat yang gemar menonton bioskop beralih kepada platform streaming film online. Media  hiburan  seperti streaming film  menjadi  hal  yang banyak diminati  dan penting  dalam kehidupan. Hal ini  dikarenakan menonton film  dapat  menghilangkan  stress  pada  seseorang. Industri penyedia jasa streaming film online seperti Netflix membutuhkan suatu algoritma untuk memberikan rekomendasi film yang sesuai dengan masing-masing pengguna. Oleh karena itu, disinilah peran machine learning dalam menciptakan sistem rekomendasi yang dapat meningkatkan performa dari jasa layanan streaming film online [[1](https://journal.uii.ac.id/AUTOMATA/article/view/17426)]

## Business Understanding

### Problem Statements

- Apa saja proses pra-pemrosesan data yang dilakukan untuk mendapatkan sistem rekomendasi yang akurat?
- Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna?

### Goals

Berdasarkan permasalahan yang dituliskan pada probelm statement, adapun tujuan dari proyek ini adalah sebagai berikut :

- Menghasilkan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering.
- Melakukan proses pra-pemrosesan data untuk mendapatkan nilai RMSE dibawah 0,25.

### Solution statements
Untuk mencapai tujuan yang ingin dicapai pada proyek ini, adapun langkah-langkah yang akan dikerjakan dalam proyek ini adalah sebagai berikut :
- **Melakukan pra-pemrosesan data**
    
    Pada proyek ini proses pra-pemrosesan data yang dilakukan meliputi :
    - menggabungkan dataset yang berbeda dataframe menjadi satu dataframe
    - mengatasi missing value pada data
    - melakukan encoding pada beberapa fitur
    - mengacak dataset 
    - melakukan normalisasi pada fitur numerik
    - melakukan pembagian data latih dan data validasi
    
- **Membangun sistem rekomendasi dengan algoritma collaborative filtering**
    
    Pada proses training, untuk menghitung kecocokan antara rating dan juga film. Dilakukan proses embending untuk data film dan juga rating. Kemudian selanjutnya dilakukan proses compile terhadap model dengan memberikan beberapa parameter. Parameter yang digunakan pada proses compiler model meliputi binary cross entropy untuk mengitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.

## Data Understanding
Dataset yang digunakan pada proyek ini merupakan kumpulan data yang dihimpun oleh MovieLens. MovieLens merupakan sebuah platform yang menyediakan jasa rekomendasi film bagi penggunanya. Dataset yang digunakan pada proyek ini merupakan sebuah dataset dari aktivitas rating dan juga tagging dari pengguna dari 9 Januari 1995- 31 Maret 2015. Akan tetapi, untuk menangani keterbatasan dalam proses komputasi, data yang akan digunakan pada proyek ini merupakan data rating film pada tahun 2015. Dataset pada proyek ini didapatkan dari platform penyedia dataset online yaitu Kaggle[2](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)

Dataset MovieLens 20M Dataset, terdiri dari 27.278 film, 20.000.263  rating dan 465.564  tag, serta terdiri dari enam file csv. Enam file tersebut meliputi :
- **movies.csv** file ini berisi tentang informasi umum dari film.
- **tag.csv** file ini berisikan tag yang ditetapkan pengguna bagi sebuah film.
- **rating.csv** file ini berisikan penilaian yang diberikan user pada sebuah film. 
- **link.csv** file ini berisi id yang digunakan oleh sumber lain untuk suatu film.
- **genome_score.csv** file ini berisikan skor relevansi setiap tag yang diberikan user dengan film.
- **genome_tags.csv** file ini berisi seluruh tag yang diberikan user pada film.

**Univariate Analysis**:

Pada tahap univariate analysis proses yang dilakukan adalah mengekplorasi data untuk mengetahui persebaran setiap fiturnya dalam dataset yang tersedia.

- **Movies** 

    variabel ini merupakan variabel dataframe yang menyimpan informasi umum dari setiap film. Informasi tersebut meliputi movieId, title atau judul film, dan genres yang menyimpan genre dari film tersebut. variabel film terdiri dari 27.278 data, dengan informasi lengkap seperti yang tertera pada tabel dibawah ini.
    
    **Tabel 1. Informasi Mengenai Variabel Movies**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | movieId | 27278	| int64 | 
    | title | 27278	| object |
    | genres | 27278	| object |

- **Tags**
    
    variabel ini terdiri dari empat fitur yaitu userId, movieId, tag dan timestamp sebagai pencatat waktu kapan tag tersebut diberikan oleh user.variabel tags terdiri dari 465.564 data, dengan informasi lengkap seperti yang tertera pada tabel dibawah ini.

    **Tabel 2. Informasi Mengenai Variabel Tags**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | userId | 465564	| int64 |
    | movieId | 465564	| int64 | 
    | tag | 465564	| object |
    | timestamp | 465564	| object |

- **Rating** 

    Variabel ratings merupakan variabel yang menunjukkan penilaian yang diberikan oleh user terhadap suatu film. Variabel ini terdiri dari empat fitur yaitu userId, movieId, rating, dan timestamp sebagai pencatat waktu ketika user memberikan penilaian terhadap sebuah film. Variabel ratings terdiri dari 20.000.263 data dengan informasi lebih lengkap seperti yang tertera pada tabel dibawah ini.
    
    **Tabel 3. Informasi Mengenai Variabel Ratings**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | userId | 20000263	| int64 |
    | movieId | 20000263	| int64 | 
    | rating | 20000263	| float64 |
    | timestamp | 20000263	| object |
    
     **Tabel 4. Distribusi Data Variabel Ratings**
    | userId | movieId | rating | 
    | ------ | ------ |------ |
    | count | 2.000026e+07 | 2.000026e+07 |	2.000026e+07 |
    | mean | 6.904587e+04 | 9.041567e+03 |	3.525529e+00
    | std	| 4.003863e+04 |	1.978948e+04 |	1.051989e+00
    | min	| 1.000000e+00 |	1.000000e+00 |	5.000000e-01
    | 25%	| 3.439500e+04 |	9.020000e+02 |	3.000000e+00
    | 50%	| 6.914100e+04 |	2.167000e+03 |	3.500000e+00
    | 75%	| 1.036370e+05 |	4.770000e+03 |	4.000000e+00
    | max	| 1.384930e+05 |	1.312620e+05 |	5.000000e+00

    Data pada tabel 4.0 menunjukkan bahwa distribusi dari rating yang diberikan merupakan rating dengan jangkauan 1 - 5.

- **links** 

    Variabel links merupakan variabel yang menunjukkan identifier yang digunakan oleh sumber lain untuk suatu film. Sumber lain dari data ini merupakan tmdb dan imdb. File ini terdiri dari tiga kolom yaitu movieId, imdbId, dan tmdbId. Variabel links terdiri dari 27278 data. Informasi lebih lanjut terkait masing-masing fitur seperti yang tampak pada tabel dibawah.

  **Tabel 5. Informasi Mengenai Variabel Links**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | movieId | 27278	| int64 | 
    | tmdbId | 27278	| int64 | 
    | imdbId | 27278	| int64 | 

- **genome_score** 

    Variabel gnome_score merupakan varibel yang menunjukkan skor relevansi setiap tag yang diberikan user dengan film. File ini terdiri dari tiga fitur yaitu movieId, tagId, dan relevance (tingkat relevansi film dengan tag yang diberikan). Variabel genome_score terdiri dari 11.709.767 data dengan informasi lebih lengkap seperti yang tertera pada tabel 6.
    
    **Tabel 6. Informasi Mengenai Variabel genome_score**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | movieId | 11709767	| int64 | 
    | tagId | 11709767	| int64 | 
    | relevance | 11709767	| float64 |

- **genome_tags** 

    variabel genome_tags merupakan variabel yang menyatakan seluruh tag yang diberikan user pada film. File ini terdiri dari dua kolom yaitu tagId dan tag. Variabel genome_tags terdiri dari 1128 data, dengan informasi lebih lengkap seperti yang tertera pada tabel 7.
    
     **Tabel 7. Informasi Mengenai Variabel genome_tags**
    | Nama Fitur | Jumlah Data Non Null | Tipe Data | 
    | ------ | ------ |------ |
    | tagId | 1128	| int64 | 
    | tag | 1128	| object |

## Data Preparation
Untuk mencapai nilai maksimum pada proses training, maka pada proyek ini dilakukan beberapa langkah pra-pemrosesan data. Langkah-langkah tersebut meliputi :

- **Menggabungkan dataset yang berbeda dataframe menjadi satu dataframe**

    Pada tahap data understanding, telah diketahui bahwasanya dataset yang tersedia terdiri dari beberapa file yang terpisah. Oleh karena itu, untuk mempermudah proses pelatihan model, serta untuk mempermudah pemahaman terhadap data, maka dilakukanlah proses penggabungan beberapa dataframe yang awalnya terpisah menjadi satu. Sehingga pada akhir tahap ini didapatkan satu dataframe yang utuh yang siap digunakan pada proses pelatihan. Untuk melakukan proses penggabungan dataframe, library pandas telah menyediakan fungsi bernama merge untuk menggabungkan dataframe.

- **Mengatasi missing value pada data**

    Untuk dapat memberikan hasil yang maksimal pada proses pelatihan, maka perlu dilakukan proses pengecekan missing value pada data yang telah digabungkan. Pada proyek ini, karena dataset yang tersedia cukup memenuhi kebutuhan dataset, maka penanganan missing value pada proyek ini dengan melakukan dropping pada baris data yang memiliki missing value. Kemudian, setelah dilakukan dropping pada missing value, dilakukan pengecekan kembali pada jenis genre film yang tersedia. Pada genre film, didapati genre film yaitu "no genre listed", untuk menangani hal tersebut dilakukan dropping pada baris data tersebut. Hal tersebut dilakukan agar semua film yang tersedia pada dataset memiliki genre yang tepat.

- **Melakukan encoding pada beberapa fitur**

    Pada proyek ini, dilakukan proses encoding pada fitur movieId dan userId. Proses encoding ini dilakukan dengan menyandikan bentuk userId dan movieId yang asli kedalam bentuk indeks integer yang berurutan. Kemudian, hasil dari proses encoding tersebut dimasukkan dalam fitur baru bernama user dan movie dan digabungkan pada dataframe yang telah tersedia. Nantinya, hasil dari encoding ini akan difungsikan pada tahap pengujian model.
    
- **Mengacak dataset**

    Sebelum kepada tahap pembagian dataset, pada proyek ini data akan diacak terlebih dahulu. Proses acak dataset ini memanfaatkan function sample dari dataframe dengan parameter random_state. Proses pengacakan ini dilakukan agar distribusi data latih maupun data pengujian menjadi tersebar secara random dan tidak terpaku pada kelompok-kelompok tertentu.

- **Melakukan normalisasi pada fitur numerik**

     Model machine learning akan memiliki performa yang baik apabila dimodelkan dengan data dengan skala yang relatif sama atau memiliki distribusi data yang normal. Untuk itu, diperlukan proses normalisasi agar data dapat berubah menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, proses normalisasi min-max scaler akan dilakukan pada fitur rating yang akan berperan sebagai label pada proses pembangunan model. Proses normalisasi min max dilakukan dengan mengurangi nilai fitur asli dengan nilai minimum lalu membaginya dengan nilai maximum yang dikurangi dengan nilai minimum. Apabila dirumuskan dengan rumus matematika, rumus dari min-max scaler adalah sebagai berikut :
     
     $$ Xi normalization = { Xi- min(X) \over max(X) - min(X) } $$

- **Melakukan pembagian data latih dan data validasi**
    
    Pada tahap ini dilakukan proses pembagian data latih dan data validasi. Data latih berperan dalam melatih model untuk menghasilkan model dengan performa yang baik. Sedangkan data validasi berperan sebagai data uji yang akan menilai sementara apakah model yang dibangun telah memenuhi kriteria yang diinginkan. Sehingga data latih ini akan menentukan apakah model siap untuk diluncurkan ataukah model masih harus mengalami evaluasi kembali.

## Modeling dan Result
Pada proyek ini, sistem rekomendasi yang digunakan menerapkan metode collaborative filtering. Collaborative Filtering merupakan sebuah metode yang didasari oleh analisis dari perilaku, aktivitas, atau selera pengguna, dan memprediksi selera pengguna berdasarkan kesamaan dengan pengguna lain. Cara ini bertujuan untuk memberikan rekomendasi kepada pengguna yang akan memilih atau menonton film tertentu berdasarkan rating yang diberikan oleh pengguna lain. Konsepnya berupa asumsi bahwa seseorang yang menyukai film tertentu juga akan menyukai film tersebut. [[3](http://jurnal.ubl.ac.id/index.php/expert/article/view/1611)]

Keluaran dari proyek ini adalah menampilkan top 5 dari film-film dengan rating tertinggi yang diberikan oleh pengguna, dan top 10 film yang direkomendasikan kepada pengguna berdasarkan rating yang diberikan oleh pengguna lain. Pada proses pembangunan model dilakukan proses perhitungan kecocokan antara pengguna dengan film dengan teknik embedding. Proses embedding dilakukan pada data user dan movie. Kemudian dilanjutkan dengan operasi perkalian dot product antara embedding user dan embedding movie. Pada model juga ditambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Pada saat melakukan kompiler pada model, terdapat beberapa parameter yang digunakan. Parameter tersebut meliputi :
- Loss Function : Binary Cross Entropy

    Loss function merupakan sebuah fungsi yang digunakan untuk menghitung perbedaan antara keluaran yang dibuat oleh algoritma saat ini dengan keluaran yang diharapkan. 
    Loss function berfungsi untuk mengevaluasi seberapa baik algoritma dalam memodelkan data dan menghasilkan keluaran yang tepat. Evaluasi akan diproses menggunakan formula matematika tertentu. Jika prediksi model mengandung banyak kesalahan, loss function akan menghasilkan angka yang lebih tinggi. Sebaliknya jika model sudah cukup bagus, maka loss function akan memberikan nilai yang lebih rendah. 

    Loss function yang digunakan pada proyek ini adalag Binary Cross Entropy. Cross Entropy digunakan saat menyesuaikan bobot model selama proses training. Tujuan dari cross entropy untuk meminimalkan loss. Karena semakin kecil loss maka semakin baik modelnya.

- Optimizer : Adam (Adaptive Moment Estimation)

    Optimizer berperan dalam mencari parameter optimal pada model untuk meminimalkan error yang terjadi. Optimizer yang digunakan pada proyek ini adalah ADAM(Adaptive Moment Estimation) Optimizer,  

- Metric Evaluation : root mean squared error (RMSE)

    Metriks Evalusi berfungsi untuk mengukur keberhasilan sebuah model dalam melakukan sebuah prediksi. Pada proyek ini, metriks evaluasi yang digunakan merupakan RMSE(Root Mean Squared Error), dimana semakin kecil nilai RMSE, maka semakin baik model yang telah dilatih.
    
    Proses pelatihan model pada proyek ini dilakukan dengan 5 epoch dengan bath size sebesar 8. Dari hasil pelatihan tersebut didapatkan hasil seperti pada tabel dibawah ini.

     **Tabel 8. Hasil Proses Pelatihan Model**
    | Epoch     | Waktu komputasi | loss | rmse | loss_val | rmse_val |
    | ------    | ------|------ | ------    |------  |------ |
    | 1         | 369 s |0.6171 |0.2193     |0.6068  | 0.2057 |
    | 2         | 381 s |0.6011 | 0.2012    | 0.6049 | 0.2032 |
    | 3         | 374 s |0.5988 | 0.1985    | 0.6042 | 0.2023 |
    | 4         | 330 s |0.5977 | 0.1971    | 0.6042 | 0.2021 |
    | 5         | 296 s |0.5965 | 0.1957    | 0.6041 | 0.2018 |
    
Pada data yang ditunjukkan pada tabel 8, dapat diketahui bahwasanya nilai loss maupun nilai rmse pada proses pelatihan maupun proses validasi cenderung menurun dari epoch pertama sampai epoch terakhir. Hal ini menunjukkan bahwasanya model yang dibangun telah menjadi tingkat good fit. 

Selanjutnya pada proses uji coba, dilakukan uji coba pada user dengan userId 135765. Dari user tersebut, berhasil didapatkan data 5 film yang mendapatkan rating tertinggi dari user 135765 dan 10 film yang belum pernah ditonton oleh user tersebut dan direkomendasikan untuk user tersebut. Untuk lebih jelasnya seperti yang tampak pada tabel 9 dan tabel 10.

**Tabel 9. Tampilan 5 film dengan rating tertinggi dari user 135765**
| Judul Film | Rating | Genre Film |
| ------    | ------|------ | 
|Hobbit: An Unexpected Journey, The (2012) |  4.5  |  Adventure, Fantasy,IMAX |
|Wolf of Wall Street, The (2013)  | 3.0  | Comedy,Crime,Drama |
|Cosmos (1980) | 4.0  | Documentary |
| Predestination (2014) | 4.0  |  Sci-Fi, Thriller |
| Before I Disappear (2014) | 4.0  |  Drama |

**Tabel 10. Tampilan 10 film rekomendasi untuk user 135765**
| Judul Film | Rating | Genre Film |
| ------    | ------|------ | 
| Blade Runner (1982) | 4.5  |  Action,Sci-Fi,Thriller |
|Good Will Hunting (1997) | 5.0  |  Drama, Romance |
|Ran (1985) | 5.0  |  Drama,War |
| Vertigo (1958) | 4.0  |  Drama,Mystery,Romance,Thriller |
|Underground (1995) | 4.5  |  Comedy,Drama,War |
|All About Eve (1950) | 5.0  |  Drama |
|Notorious (1946) | 5.0  |  Film-Noir,Romance,Thriller |
|2001: A Space Odyssey (1968) | 2.5  |  Adventure, Drama, Sci-Fi |
|Treasure of the Sierra Madre, The (1948) | 0.5  |  Action, Adventure, Drama, Western |
|400 Blows, The (Les quatre cents coups) (1959) | 3.5  |  Crime, Drama |

## Evaluation
Untuk mengitung performa dari suatu model, diperlukan adanya metrik evaluasi. Pada proyek ini, metriks evaluasi yang digunakan merupakan RMSE (Root Mean Squared Error). Proyek ini menggunakan metrik evaluasi RMSE dikarenakan proyek ini termasuk kedalam sistem rekomendasi yang masuk kedalam ranah regresi. Regresi adalah proses identifikasi relasi dan pengaruhnya pada nilai-nilai objek. Regresi bertujuan untuk menemukan sutau fungsi yang memodelkan data dengan meminimalisir galat atau selisih antra nilai prediksi dengan nilai sebenarnya. Untuk menghitung performa dari suatu model regresi, RMSE adalah metriks evaluasi yang tepat.

RMSE dihitung dengan mencari nilai akar dari sigma kuadrat nilai prediksi yang dikurangi dengan nilai sesungguhnya dan dibagi dengan jumlah data. Apabila dirumusakan dalam rumusan matematika, akan tampak seperti persamaan dibawah ini.
![image](https://user-images.githubusercontent.com/67575741/197354045-dd427686-7ee0-4ba9-bc11-8121a434ae53.png)

Keterangan :

Y ' = Nilai Prediksi

Y   = Nilai Sejati

n    = Jumlah Data

Dari persamaan diatas, didapatkan nilai rmse pada proses pelatihan dan pengujian seperti yang tampak pada grafik dibawah ini.

**Gambar 1. Grafik Root Mean Squared Error Model**
![download (1)](https://user-images.githubusercontent.com/67575741/197353938-718632d4-1760-4c7c-a652-3de2980010ee.png)

Dari grafik yang ditunjukkan oleh gambar 1, dapat dilihat bahwasanya nilai rmse baik pada proses training maupun validation terjadi penurunan pada setiap epochnya. Hal ini menunjukkan bahwasanya model yang dibangun termasuk kedalam kategori good fit. Selain itu, model ini telah memenuhi target untuk memiliki rmse dibawah 0,25. Karena pada epoch terkahir dapat dilihat nilai rmse dibawah 0,2.


# Referensi
[1] M. Rizqi Az Zayyad and A. Kurniawardhani, "Penerapan Metode Deep Learning pada Sistem Rekomendasi Film", journal.uii.ac.id, vol. 2, no. 1, p. 5, 2021. [Accessed 13 October 2022].

[2] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

[3] Erlangga, E., & Sutrisno, H. (2020). Sistem Rekomendasi Beauty Shop Berbasis Collaborative Filtering. EXPERT: Jurnal Manajemen Sistem Informasi Dan Teknologi, 10(2), 47-52.


