#!/usr/bin/env python
# coding: utf-8

# # Data Information
# 
# ## Atribut Information
# Belakangan ini banyak sekali penyebaran informasi atau berita hoak di internet. Dampaknya, masyarakat akan merasa yakin bahwa konten tersebut benar tanpa ada unsur kebohongan sama sekali. Selain itu, dampak lain dari hoax adalah bisa merugikan emosi hingga finansial masyarakat. Dikutip dari laman Psychology Today, target dari penyebaran hoax ini adalah emosi mereka yang membacanya.
# 
# Jadi pada kasus kali ini kita akan membuat model untuk mendeteksi apakah suatu berita Real/Fake. Dengan adanya ini kita bisa memfilter mana berita yang hoak dan berita yang real, sehingga informasi yang di konsumsi oleh pengguna internet adalah berita yang asli/real.
# 
# Dataset hanya terdiri dari 4 kolom yaitu,
# * number: hanya nummber biasa.
# * title: Judul dari informasi.
# * text: isi dari informasi.
# * label: label REAL/FAKE.
# 
# Link: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

# ## Import Library

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
get_ipython().system('pip install nltk')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# ## Import dataset

# In[3]:


df = pd.read_csv('fake_or_real_news.csv')
df.head()


# Setelah kita import dataset, ia memiliki 4 kolom. Namun kali ini pada kasus kita, kita hanya butuh 2 kolom yaitu kolom `title` dan `label`. Sisanya kita bisa drop dengan fungsi `drop()`

# In[3]:


df.drop(['Unnamed: 0', 'title'], axis=1, inplace=True)
df.head()


# Sekarang data kita sudah menjadi dua kolom/atribut. Kemudian kita langsung ke tahapa Data Preparation untuk memudahkan proses kali ini, karena biasanya kita melakukan EDA(Exploratory Data Analysis) untuk mengenal lebih tentang data kita. Mungkin pada project selanjutnya akan kita buat secara lengkap prosesnya.

# # Data Preparation
# 
# ## Stopword
# Stopword merupakan dimana kita akan mengabaikan kata yang memiliki sedikit makna semisal 'a', 'the', 'is' dll. Tujuan utama dalam penerapan proses stopwords ini adalah mengurangi jumlah kata dalam sebuah dokumen yang nantinya akan berpengaruh dalam kecepatan dan performa NLP.

# In[4]:


# Stopword
df['text'] = df['text'].str.lower()
stopwords = stopwords.words('english')
df['text'] = df['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stopwords)]))


# In[5]:


df.head()


# Sekarang kita lihat bahwa data `text` sudah bersih dari kata-kata yang memiliki sedikit makna, dan mengubah ke semua huruf menjadi lower atau huruf kecil.

# ## Labeling
# Tahap ini kita akan mengubah label kita menjadi numerik, karena mesin tidak bisa membaca text/string, oleh karena itu kita akan mengubahnya dan mempresentasikan, 
# * REAL : 1
# * FAKE : 0
# 
# Dengan begitu mesin akan bisa mengetahui isi dari data label kita, dengan menggunakan `LabelEncoder` dari library scikit learn kita bisa dengan mudah melakukan labeling terhadap data kita.

# In[6]:


# Labeling Label dataset
label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
df.head()


# ## Split Data into Training and Test set
# Kemudian kita akan membagi data kita menjadi data latih dan data uji, jadi yang akan kita training adalah data latih, dan pada tahap percobaan hasil latih kita menggunakan data test. Kita membagi data kita sebesar 10% dari total data yang kita miliki, dengan scikit learn kita bisa membagi data kita dengan mudah.

# In[7]:


# Split data into training and test sets
sentence = df['text'].values
label = df['label'].values

sentence_train, sentence_test, label_train, label_test = train_test_split(sentence, label, test_size=0.1, random_state=42)


# ## Tokenize
# Sebelumnya kita tau bahwa mesin tidak bisa memproses text/string, oleh karena itu kita juga akan mengkonfersi setiap kata menjadi angka numerik. Salah satu caranya adalah dengan tokenisasi. Kita menggunakan tokenizer dari Tensorflow, untuk mengkonversikannya.
# 
# Konsepnya adalah kita akan memisahkan sebuah kalimat menjadi kata dan mempresentasikan sebuah kata tersebut menjadi numerik, contoh.<br />
# "I Love my mother"  ==  [1, 2, 3, 4]
# * I = 1
# * Love = 2
# * my = 3
# * mother = 4
# 
# Kita juga akan menfilter simbol-simbol yang dapat mengganggu pada saat training seperti `!"#$%&()*+.,-/:;=?@[\]^_{|}~` <br />
# Dan terakhir kita mengatasi kata-kata yang ambigu atau tidak mengerti apa artinya itu, dengan `<OOV>` kita merepresentasikan kata tersebut menjadi angka `1`, setelah ini kita akan melihatnya jadi langsung ajah.
# 

# In[8]:


# Tokenize
filt = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~'
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>", filters=filt) # <OOV> untuk mengatasi kata yang tidak ada di vocab/tidak di ketahui
tokenizer.fit_on_texts(sentence_train) # Sequences
tokenizer.fit_on_texts(sentence_test) # Sequences


# Setelah ini kita akan mencoba melihat hasil dari tokenisasi kita, dan menjadi gambaran seperti apa toknisasi menggunakan tensorflow

# In[9]:


# Untuk melihat Indek kata
word_index = tokenizer.word_index
print(word_index)


# In[10]:


# Untuk melihat sequence (kalimat)
sequences = tokenizer.texts_to_sequences(sentence_train)
print(sequences)


# ## Sequens and Padding
# Sequense sebenarnya sudah kita lakukan pada code sebelumnya untuk melihat sequens (kalimat), oke tapi akan saya coba jelaskan.
# 
# **Sequence**: Merupakan kumpulan larik angka dari beberapa kata.<br/>
# **Padding**: proses untuk membuat setiap kalimat pada teks memiliki panjang yang seragam.
# 
# Contoh:
# 
# === **Sequence** ===
# 
# "Aku sayang ibu" == [1, 2, 3]<br/>
# "Aku sayang kakak" == [1, 2, 4]<br/>
# "Aku sayang ayah" == [1, 2, 5]
# 
# Larik angka tersebut di sebut dengan Sequence, jadi kita bisa simpulkan dengan bahasa kita sendiri bahwa sequence adalah larik angka dari beberapa kata yang sudah di tokenize.
# 
# === **Padding** ===
# 
# "Aku sayang ibu" == [1, 2, 3]<br/>
# "Aku sayang kakak" == [1, 2, 4]<br/>
# "Aku sayang paman dan bibi" == [1, 2, 5, 6, 7]
# 
# dengan padding kita akan menyelaraskan larik tersebut, bisa dengan menyamakan panjang larik dari larik terpanjang, atau kita bisa menentukannya sendiri. Contoh kita kali ini akan menyamakan panjang larik dari larik yang terpanjang, maka akan menjadi seperti ini.
# 
# [1, 2, 3, 0, 0]<br/>
# [1, 2, 4, 0, 0]<br/>
# [1, 2, 5, 6, 7]
# 
# Karena kedua larik di atas mengikuti larik terpanjang, maka agar sama panjang ditambahlah angka `0` di sana. kita juga bisa menentukan angka nol tersebut mau di taruh depan atau di belakang sama seperti contoh di atas.

# In[11]:


# membuat Sequens
train_sekuens = tokenizer.texts_to_sequences(sentence_train)
test_sekuens = tokenizer.texts_to_sequences(sentence_test)

# membuat padding
train_padded = pad_sequences(train_sekuens, padding='post', truncating='post')
test_padded = pad_sequences(test_sekuens, padding='post', truncating='post')


# # Modeling
# Kita akan menggunakan tensorflow untuk melakukan proses modeling, dengan memanfaatkan Sequential kita dengan mudah akan mengatur Embedding, dan layar Dense, kita juga menggunakan GlobalAverangePooling1D.

# In[12]:


# membuat model
model = Sequential([
    tf.keras.layers.Embedding(2000, 20, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[13]:


# compile model
model.compile(optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy'])


# In[14]:


model.summary()


# Code di bawah adalah dimana proses training di mulai, kita menentukan epoch sebanyak `30`. Epoch adalah perulangan dalam proses latih, gampangnya seperti kita berlari 30 putaran di lapangan. Epoch 30 menurut saya sudah cukup bagus, karena sebelumnya saya sudah melatih lebih dari ini dan hasilnya tidak beda jauh, bahkan terkadang menjadi lebih buruk.

# In[15]:


# Train model
num_epochs = 30
history = model.fit(train_padded, label_train,
                    epochs=num_epochs,
                    validation_data=(test_padded, label_test),
                    verbose=2)


# # Visualization of Training Results
# Tahap ini kita akan melihat hasil training kita dengan visualisasi agar mudah di pahami.

# In[16]:


# visualisasi training

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# # Test the Model

# In[18]:


predictions = model.predict(test_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]


print(test_padded[10:15])

print(f'\nLabel Aktual: {label_test[10:15]}')
print(f'Label Prediksi: {predictions[10:15]}')


# # Kesimpulan
# Kalau kalian lihat di atas, bahwa kita memiliki prediksi yang bagus. Kita mengatur untuk **test_padded** dengan index 10 sampai 15 yaitu dimana itu adalah hasil split data test untuk `text` yang sudah di padding.<br/>
# maka ketika di outputkan seperti ini:<br/>
# 
# [[610 475 102 ...   0   0   0]<br/>
# [  5 534   1 ...   0   0   0]<br/>
# [ 39 700 343 ...   0   0   0]<br/>
# [ 95 186   1 ...   0   0   0]<br/>
# [934   1 326 ...   0   0   0]]
# 
# 
# Untuk mengetahui label dari **test_padded** index 10-15, kita memanggil **label_test** yang dimana itu adalah label aktual dari **test_padded**. Dan kita mendapati [0 1 1 0 0] sebagai nilai aktual, simpelnya itu adalah label asli dari dataset kita.
# 
# Kemudian kita memprediksi hasil dari **test_padded** menggunakan predections, dan di dapati angka [1 1 1 0 0]
# 
# Ingat sebelumnya:<br/>
# * REAL : 1
# * FAKE : 0
# 
# Jika kita perhatikan:<br/>
# Label Aktual   : [0, 1, 1, 0, 0]<br/>
# Label Prediksi : [1, 1, 1, 0, 0]
# 
# prediksi kita sebagian besar benar, dan hanya satu yang memiliki kesalahan prediksi, yaitu pada index `10` yang di mana nilai sebenarnya adalah `0(FAKE)` tapi program kita memprediksi itu adalah `1(REAL)`. Tapi tidak apa-apa, karna akurasi yang kita raih sudah cukup bagus sebesar `96%` untuk **akurasi** dan `93%` untuk **akurasi validasi**.
# 
# Jadi kalian bisa eksperimen dengan mengganti indexnya dan melihat hasil prediksi dari model kita, jadi itu cara membuat model `NLP` sederhana dengan `Tensorflow`. Terima Kasih!!!
# 

# In[ ]:




