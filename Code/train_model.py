import os
import pandas as pd
import re
import joblib
import scipy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# ==== Step 0: Define Paths ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ==== Step 1: Load CSVs ====
df1 = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_dark_patterns_labeled.csv'))
df2 = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_dataux_labeled.csv'))
df_keywords = pd.read_csv(os.path.join(DATA_DIR, 'dark_pattern keyword.csv'))

print("📌 df1 columns:", df1.columns.tolist())
print("📌 df2 columns:", df2.columns.tolist())

# ==== Step 2: Standardize Columns ====
# Convert multiple UI fields into a single 'text' field
df1['text'] = df1[['Buttons', 'Links', 'Checkboxes', 'Popups']].astype(str).agg(' '.join, axis=1)
df1['dark_pattern_label'] = 'unknown'  # Set a placeholder or assign proper label if known

# Keep only required columns
df1 = df1[['text', 'dark_pattern_label']]
df2 = df2[['text', 'dark_pattern_label']]

# ==== Step 3: Combine & Drop NA ====
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['text', 'dark_pattern_label'], inplace=True)

# ==== Step 4: Clean Text ====
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    return text.lower()

df['cleaned_text'] = df['text'].astype(str).apply(clean_text)

# ==== Step 5: Keyword Matching ====
df_keywords['keyword'] = df_keywords['keyword'].astype(str).str.lower()
keyword_list = df_keywords['keyword'].tolist()

def keyword_flag(text):
    return int(any(k in text for k in keyword_list))

df['has_keyword'] = df['cleaned_text'].apply(keyword_flag)

# ==== Step 6: Encode Labels ====
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['dark_pattern_label'])

# ==== Step 7: TF-IDF + Feature Merge ====
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['cleaned_text'])
X = scipy.sparse.hstack([X_text, df[['has_keyword']].values])
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Step 8: Train Model ====
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==== Step 9: Evaluate ====
pred_labels = clf.predict(X_test)
valid_labels = unique_labels(y_test, pred_labels)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, pred_labels, labels=valid_labels, target_names=le.classes_[valid_labels]))

# ==== Step 10: Save Artifacts ====
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, os.path.join(MODEL_DIR, 'dark_pattern_detector.pkl'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("\n✅ Model training complete and saved to 'models/' folder.")
