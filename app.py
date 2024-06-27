from flask import Flask, render_template, request
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Memuat dan memproses dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
nama_kolom = ["Sequence Name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Class"]
df = pd.read_csv(url, delim_whitespace=True, names=nama_kolom)

# Memisahkan fitur dan label
kolom_fitur = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
X = df[kolom_fitur].values
y = df["Class"].values

# Encode label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih classifier SVM
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Mendefinisikan rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Mendefinisikan rute untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mendapatkan data input dari form
        mcg = float(request.form['mcg'])
        gvh = float(request.form['gvh'])
        lip = float(request.form['lip'])
        chg = float(request.form['chg'])
        aac = float(request.form['aac'])
        alm1 = float(request.form['alm1'])
        alm2 = float(request.form['alm2'])

        # Membuat prediksi untuk data input
        x_new = [[mcg, gvh, lip, chg, aac, alm1, alm2]]
        x_new = scaler.transform(x_new)
        prediction = classifier.predict(x_new)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        train_prediction = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_prediction)

        test_prediction = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_prediction)

    return render_template('predict.html', prediction=predicted_class, train_accuracy=train_accuracy, test_accuracy=test_accuracy)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
