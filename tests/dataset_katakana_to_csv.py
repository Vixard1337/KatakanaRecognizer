import os
import pandas as pd

# Lista wszystkich znaków Katakana
katakana_characters = ['ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
                  'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
                  'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
                  'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ',
                  'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン']

# Tworzenie DataFrame
try:
    df = pd.DataFrame({'character': katakana_characters})

    # Upewnij się, że katalogi istnieją
    os.makedirs('data/raw', exist_ok=True)

    # Zapis do pliku CSV
    df.to_csv('data/raw/katakana_labels.csv', index=False, encoding='utf-8-sig')
    print("Plik CSV został zapisany pomyślnie.")
except Exception as e:
    print(f"Wystąpił błąd: {e}")
    
# Odczyt zawartości pliku CSV
with open('data/raw/katakana_labels.csv', 'r', encoding='utf-8-sig') as f:
    content = f.read()
    print("Zawartość pliku CSV:")
    print(content)
    
    # Wczytaj plik CSV
df = pd.read_csv('data/raw/katakana_labels.csv')

# Sprawdź liczbę wierszy i pierwsze kilka wierszy
print(f"Liczba wierszy: {len(df)}")
print(df.head(46))  # Zobacz wszystkie znaki, jeśli ich jest 46

