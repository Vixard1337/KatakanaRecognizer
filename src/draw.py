import io
import os
import numpy as np
import cv2
import tensorflow as tf
import glob
import time
import customtkinter as ctk
import tkinter.messagebox as messagebox
from model import create_model
from PIL import Image, ImageDraw, ImageTk

ctk.set_appearance_mode("System")  # Options: "System" (default), "Light", "Dark"
ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue")

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, label_status, label_time, label_accuracy, epochs, text_widget, steps_per_epoch):
        super().__init__()
        self.progress_bar = progress_bar
        self.label_status = label_status
        self.label_time = label_time
        self.label_accuracy = label_accuracy
        self.epochs = epochs
        self.start_time = time.time()
        self.text_widget = text_widget
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1
        progress = self.current_step / self.steps_per_epoch
        self.progress_bar.set(progress)
        self.text_widget.insert(ctk.END, f"Batch {self.current_step}/{self.steps_per_epoch} - loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.2f}%\n")
        self.text_widget.see(ctk.END)
        
        # Aktualizacja GUI
        self.progress_bar.update_idletasks()
        self.text_widget.update_idletasks()

    def on_epoch_end(self, epoch, logs=None):
        self.current_step = 0
        current_time = time.time() - self.start_time
        self.label_status.configure(text=f"Epoka {epoch + 1} z {self.epochs}")
        self.label_time.configure(text=f"Czas trwania: {current_time:.2f} sekundy")
        accuracy = logs.get("accuracy", 0) * 100
        self.label_accuracy.configure(text=f"Dokładność: {accuracy:.2f}%")
        self.text_widget.insert(ctk.END, f"Epoka {epoch + 1}/{self.epochs} zakończona - loss: {logs.get('loss'):.4f} - accuracy: {accuracy:.2f}%\n")
        self.text_widget.see(ctk.END)
        
        # Aktualizacja GUI
        self.label_status.update_idletasks()
        self.label_time.update_idletasks()
        self.label_accuracy.update_idletasks()
        self.text_widget.update_idletasks()
        
    def on_train_end(self, logs=None):
        self.label_status.configure(text="Trenowanie zakończone!")
        self.label_time.configure(text=f"Czas całkowity: {time.time() - self.start_time:.2f} sekundy")
        final_accuracy = logs.get("accuracy", 0) * 100
        self.label_accuracy.configure(text=f"Ostateczna dokładność: {final_accuracy:.2f}%")
        self.text_widget.insert(ctk.END, "Trenowanie zakończone!\n")
        self.text_widget.see(ctk.END)
        
        # Aktualizacja GUI
        self.label_status.update_idletasks()
        self.label_time.update_idletasks()
        self.label_accuracy.update_idletasks()
        self.text_widget.update_idletasks()
        
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event):
        if self.tip_window or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = ctk.CTkLabel(tw, text=self.text, justify='left', wraplength=200)
        label.pack(ipadx=1)

    def hide_tip(self, event):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

data_dir = '/home/admin/Dyplom/katakana/data/raw/google_fonts/Kosugi/Regular'

class KatakanaRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Katakana Recognizer")
        self.root.geometry("800x600")

        # Initialize network parameters
        self.filters = ctk.IntVar(value=64)
        self.kernel_size = ctk.IntVar(value=3)
        self.epochs = ctk.IntVar(value=20)
        self.layers = ctk.IntVar(value=3)

        # Drawing parameters
        self.canvas_size = 200
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # Drawing area
        self.canvas = ctk.CTkCanvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.clear_button = ctk.CTkButton(self.root, text="Wyczyść", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.recognize_button = ctk.CTkButton(self.root, text="Rozpoznaj", command=self.recognize_character)
        self.recognize_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        self.upload_button = ctk.CTkButton(self.root, text="Edytor graficzny", command=self.open_upload_window)
        self.upload_button.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        # Network settings
        ctk.CTkLabel(self.root, text="Liczba filtrów:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.filters_entry = ctk.CTkEntry(self.root, textvariable=self.filters)
        self.filters_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        ToolTip(self.filters_entry, "Liczba filtrów: Określa liczbę filtrów w warstwach konwolucyjnych, wpływając na zdolność sieci do wykrywania cech obrazu.")

        ctk.CTkLabel(self.root, text="Rozmiar jądra:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.kernel_size_entry = ctk.CTkEntry(self.root, textvariable=self.kernel_size)
        self.kernel_size_entry.grid(row=3, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        ToolTip(self.kernel_size_entry, "Rozmiar jądra: Ustawia wielkość okna konwolucyjnego, decydując o obszarze analizowanym na obrazie przez każdy filtr.")

        ctk.CTkLabel(self.root, text="Liczba warstw:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.layers_entry = ctk.CTkEntry(self.root, textvariable=self.layers)
        self.layers_entry.grid(row=4, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        ToolTip(self.layers_entry, "Liczba warstw: Kontroluje głębokość sieci, co wpływa na zdolność rozpoznawania złożonych wzorców.")

        ctk.CTkLabel(self.root, text="Epoki:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.epochs_entry = ctk.CTkEntry(self.root, textvariable=self.epochs)
        self.epochs_entry.grid(row=5, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        ToolTip(self.epochs_entry, "Epoki: Określa liczbę pełnych przejść przez dane treningowe, wpływając na czas i jakość uczenia się modelu.")

        
        def validate_parameters():
            try:
                filters = int(self.filters.get())
                kernel_size = int(self.kernel_size.get())
                layers = int(self.layers.get())
                epochs = int(self.epochs.get())
               
                if not (32 <= filters <= 256):
                    raise ValueError("Liczba filtrów musi być w zakresie [32, 256].")
                if not (2 <= kernel_size <= 5):
                    raise ValueError("Rozmiar jądra musi być w zakresie [2, 5].")
                if not (2 <= layers <= 6):
                    raise ValueError("Liczba warstw musi być w zakresie [2, 6].")
                if not (10 <= epochs <= 50):
                    raise ValueError("Liczba epok musi być w zakresie [10, 50].")
                    
                return True
            except ValueError as e:
                messagebox.showerror("Błąd walidacji", str(e))
                return False

        def on_train_button_click():
                    if validate_parameters():
                        self.train_model()
                        pass

        # Training button
        self.train_button = ctk.CTkButton(self.root, text="Trenuj model", command=on_train_button_click)
        self.train_button.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        self.gallery_button = ctk.CTkButton(self.root, text="Galeria Katakana", command=self.show_gallery)
        self.gallery_button.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        self.weights_button = ctk.CTkButton(self.root, text="Wyświetl wagi neuronów", command=self.show_weights)
        self.weights_button.grid(row=8, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Configure grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black", width=5)
        self.draw.ellipse([x-2, y-2, x+2, y+2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        
    def preprocess_image(self):
            canvas_image = self.canvas.postscript(colormode='color')
            image = Image.open(io.BytesIO(canvas_image.encode('utf-8')))
            image = image.convert('L')  # Konwertuj na skalę szarości
            image = image.resize((100, 100), Image.LANCZOS)  # Zmień rozmiar na 28x28
            img_array = np.array(image)
            img_array = img_array / 255.0  # Normalizacja
            img_array = img_array.reshape(1, 100, 100, 1)  # Dopasowanie do wymagań modelu
            return img_array
        
    def show_weights(self):
        weights_window = ctk.CTkToplevel(self.root)
        weights_window.title("Wagi modelu")
        weights_window.geometry("600x600")

        # Dodanie widgetu tekstowego do wyświetlania wag
        text_widget = ctk.CTkTextbox(weights_window, width=580, height=580)
        text_widget.pack(pady=10)

        # Pobranie parametrów modelu
        filters = self.filters.get()
        kernel_size = self.kernel_size.get()
        epochs = self.epochs.get()
        layers = self.layers.get()
        input_shape = (100, 100, 1)
        num_classes = 46
        weights_path = 'weights.h5'

        # Tworzenie modelu
        model = create_model(input_shape, num_classes, filters, kernel_size, layers)
        learning_rate = 0.001  # Współczynnik uczenia
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        try:
            model.load_weights(weights_path)
            text_widget.insert(ctk.END, "Wagi zostały załadowane.\n")
        except:
            text_widget.insert(ctk.END, "Nie udało się załadować wag.\n")
            return

        # Wyświetlanie parametrów sieci
        text_widget.insert(ctk.END, f"Parametry modelu:\n")
        text_widget.insert(ctk.END, f"Filtry: {filters}\n")
        text_widget.insert(ctk.END, f"Rozmiar jądra: {kernel_size}\n")
        text_widget.insert(ctk.END, f"Liczba warstw: {layers}\n")
        text_widget.insert(ctk.END, f"Liczba epok: {epochs}\n")
        text_widget.insert(ctk.END, f"Współczynnik uczenia: {learning_rate}\n")
        text_widget.insert(ctk.END, f"Optymalizator: Adam\n\n")

        # Wyświetlanie wag z poszczególnych warstw
        for layer in model.layers:
            text_widget.insert(ctk.END, f"Warstwa: {layer.name}\n")
            weights = layer.get_weights()
            if weights:
                for i, weight in enumerate(weights):
                    text_widget.insert(ctk.END, f"Waga {i+1}:\n{weight}\n\n")
            else:
                text_widget.insert(ctk.END, "Brak wag do wyświetlenia.\n\n")

        text_widget.insert(ctk.END, "Wyświetlanie wag zakończone.\n")
    def recognize_character(self):
        start_time = time.time() 
        img_array = self.preprocess_image()
        model = tf.keras.models.load_model('katakana_cnn_model.h5')
        predictions = model.predict(img_array)
        top_5 = tf.nn.top_k(predictions, k=5)
        top_5_classes = top_5.indices.numpy()[0]
        top_5_probabilities = (top_5.values.numpy()[0] * 100).tolist()
        end_time = time.time()
        prediction_time = end_time - start_time
        self.show_result(top_5_classes, top_5_probabilities, prediction_time)

    def show_result(self, top_5_classes, top_5_probabilities, prediction_time):
        result_window = ctk.CTkToplevel(self.root)
        result_window.title("Wynik rozpoznania")

        class_to_char = {
            0: 'ア', 1: 'イ', 2: 'ウ', 3: 'エ', 4: 'オ',
            5: 'カ', 6: 'キ', 7: 'ク', 8: 'ケ', 9: 'コ',
            10: 'サ', 11: 'シ', 12: 'ス', 13: 'セ', 14: 'ソ',
            15: 'タ', 16: 'チ', 17: 'ツ', 18: 'テ', 19: 'ト',
            20: 'ナ', 21: 'ニ', 22: 'ヌ', 23: 'ネ', 24: 'ノ',
            25: 'ハ', 26: 'ヒ', 27: 'フ', 28: 'ヘ', 29: 'ホ',
            30: 'マ', 31: 'ミ', 32: 'ム', 33: 'メ', 34: 'モ',
            35: 'ヤ', 36: 'ユ', 37: 'ヨ',
            38: 'ラ', 39: 'リ', 40: 'ル', 41: 'レ', 42: 'ロ',
            43: 'ワ', 44: 'ヲ', 45: 'ン'
        }

        for i in range(5):
            recognized_char = class_to_char.get(top_5_classes[i], "Nieznany znak")
            probability = top_5_probabilities[i]
            ctk.CTkLabel(result_window, text=f"Znak: {recognized_char}, Prawdopodobieństwo: {probability:.2f}%").pack(padx=20, pady=10)

        recognized_char = class_to_char.get(top_5_classes[0], "Nieznany znak")
        images_dir = '/home/admin/Dyplom/katakana/data/raw/organized'
        image_path_pattern = os.path.join(images_dir, recognized_char, '*Kosugi_Regular*.jpg')
        image_files = glob.glob(image_path_pattern)

        if image_files:
            img = Image.open(image_files[0])
            img = img.resize((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            img_label = ctk.CTkLabel(result_window, image=img_tk, text="")  # Set text to empty
            img_label.image = img_tk
            img_label.pack(padx=20, pady=10)
        else:
            ctk.CTkLabel(result_window, text="Nie znaleziono obrazu dla rozpoznanego znaku.").pack(padx=20, pady=10)

        # Display additional information
        ctk.CTkLabel(result_window, text=f"Czas predykcji: {prediction_time:.2f} sekundy").pack(padx=20, pady=10)
        ctk.CTkLabel(result_window, text=f"Rozpoznany znak: {recognized_char}").pack(padx=20, pady=10)
    
    def train_model(self):
        # Tworzenie nowego okna do wyświetlania postępu trenowania
        progress_window = ctk.CTkToplevel(self.root)
        progress_window.title("Postęp trenowania")
        progress_window.geometry("400x400")

        # Dodanie paska postępu
        progress_bar = ctk.CTkProgressBar(progress_window, width=300)
        progress_bar.set(0)
        progress_bar.pack(pady=10)

        # Dodanie etykiet statusu
        label_status = ctk.CTkLabel(progress_window, text="Rozpoczęto trenowanie...", fg_color="blue", text_color="white")
        label_status.pack(pady=5)
        label_time = ctk.CTkLabel(progress_window, text="Czas trwania: 0.0 sekundy", fg_color="blue", text_color="white")
        label_time.pack(pady=5)
        label_accuracy = ctk.CTkLabel(progress_window, text="Dokładność: 0.0%", fg_color="blue", text_color="white")
        label_accuracy.pack(pady=5)

        # Dodanie widgetu tekstowego do wyświetlania logów
        text_widget = ctk.CTkTextbox(progress_window, width=380, height=200)
        text_widget.pack(pady=10)

        # Pobranie parametrów modelu
        filters = self.filters.get()
        kernel_size = self.kernel_size.get()
        epochs = self.epochs.get()
        layers = self.layers.get()
        input_shape = (100, 100, 1)
        num_classes = 46
        weights_path = 'weights.h5'

        # Tworzenie modelu
        model = create_model(input_shape, num_classes, filters, kernel_size, layers)
        learning_rate = 0.001  # Współczynnik uczenia
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Wypisanie parametrów sieci i współczynnika uczenia
        text_widget.insert(ctk.END, f"Parametry modelu:\n")
        text_widget.insert(ctk.END, f"Filtry: {filters}\n")
        text_widget.insert(ctk.END, f"Rozmiar jądra: {kernel_size}\n")
        text_widget.insert(ctk.END, f"Liczba warstw: {layers}\n")
        text_widget.insert(ctk.END, f"Liczba epok: {epochs}\n")
        text_widget.insert(ctk.END, f"Współczynnik uczenia: {learning_rate}\n")
        text_widget.insert(ctk.END, f"Optymalizator: Adam\n")

        # Ustawienie generatorów danych
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
        data_dir = '/home/admin/Dyplom/katakana/data/processed/augmented'
        train_generator = train_datagen.flow_from_directory(
            data_dir, target_size=(100, 100), color_mode='grayscale', batch_size=32,
            class_mode='categorical', subset='training')
        validation_generator = train_datagen.flow_from_directory(
            data_dir, target_size=(100, 100), color_mode='grayscale', batch_size=32,
            class_mode='categorical', subset='validation')

        steps_per_epoch = len(train_generator)

        # Callbacki
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_katakana_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        progress_callback = TrainingProgressCallback(progress_bar, label_status, label_time, label_accuracy, epochs, text_widget, steps_per_epoch)

        try:
            model.load_weights(weights_path)
            text_widget.insert(ctk.END, "Wagi zostały załadowane.\n")
        except:
            text_widget.insert(ctk.END, "Nie udało się załadować wag, trenowanie od początku.\n")

        # Rozpoczęcie trenowania z callbackiem
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, progress_callback]
        )

        # Zapisanie modelu
        model.save_weights(weights_path)
        text_widget.insert(ctk.END, "Wagi zostały zapisane.\n")
        model.save('katakana_cnn_model_custom.h5')
        label_status.configure(text="Model zapisany jako katakana_cnn_model_custom.h5")
    
    def show_gallery(self, event=None):
        # Tworzenie nowego okna galerii i ustawienie większego rozmiaru
        gallery_window = ctk.CTkToplevel(self.root)
        gallery_window.title("Galeria Katakana")
        gallery_window.geometry("800x600")  # Ustawienie większego rozmiaru okna

        # Tworzenie przewijalnej ramki
        scrollable_frame = ctk.CTkScrollableFrame(gallery_window)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Konfiguracja siatki galerii
        for i in range(5):
            scrollable_frame.columnconfigure(i, weight=1)

        # Mapa znaków Katakana na ich wymowę
        katakana_pronunciation = {
            'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
            'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
            'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
            'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
            'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
            'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
            'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
            'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
            'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
            'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n'
        }

        # Pobranie listy plików obrazów
        image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Wyświetlanie obrazów i podpisów z numeracją
        for i, image_file in enumerate(image_files):
            img_path = os.path.join(data_dir, image_file)
            img = Image.open(img_path)
            img = img.resize((100, 100), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)

            # Etykieta dla obrazu
            img_label = ctk.CTkLabel(scrollable_frame, image=img, text="")  # Tekst pusty, aby uniknąć nakładania
            img_label.image = img  # Przechowywanie referencji, aby zapobiec usunięciu przez garbage collector
            img_label.grid(row=i // 5 * 2, column=i % 5, padx=10, pady=10)

            # Wyciąganie nazwy znaku z nazwy pliku i dodanie wymowy wraz z numeracją
            char_name = os.path.splitext(image_file)[0]
            pronunciation = katakana_pronunciation.get(char_name, "Nieznana wymowa")
            char_label = ctk.CTkLabel(scrollable_frame, text=f"{i + 1}. {char_name} ({pronunciation})")
            char_label.grid(row=i // 5 * 2 + 1, column=i % 5, padx=10, pady=10)
            
    def open_upload_window(self):
        upload_window = ctk.CTkToplevel(self.root)
        upload_window.title("Edytor obrazu")
        upload_window.geometry("600x600")

        self.uploaded_image_label = ctk.CTkLabel(upload_window, text="Brak obrazu")
        self.uploaded_image_label.pack(padx=10, pady=10)

        # Capture the canvas image and display it in the editor
        self.capture_canvas_image()
        self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
        self.uploaded_image_label.configure(image=self.uploaded_image_tk, text="")
        self.history = [self.uploaded_image.copy()]

        # Create a frame for the sliders and buttons
        control_frame = ctk.CTkFrame(upload_window)
        control_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Rotate controls
        rotate_label = ctk.CTkLabel(control_frame, text="Obrót (stopnie)")
        rotate_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rotate_slider = ctk.CTkSlider(control_frame, from_=-30, to=30, number_of_steps=60, command=self.update_rotate_label)
        self.rotate_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.rotate_value_label = ctk.CTkLabel(control_frame, text="0°")
        self.rotate_value_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        rotate_button = ctk.CTkButton(control_frame, text="Dodaj Obrót", command=self.rotate_image)
        rotate_button.grid(row=0, column=3, padx=5, pady=5)

        # Shear controls
        shear_label = ctk.CTkLabel(control_frame, text="Ścinanie (procent)")
        shear_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.shear_slider = ctk.CTkSlider(control_frame, from_=0.2, to=0.5, number_of_steps=30, command=self.update_shear_label)
        self.shear_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.shear_value_label = ctk.CTkLabel(control_frame, text="20%")
        self.shear_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        shear_button = ctk.CTkButton(control_frame, text="Dodaj Ścinanie", command=self.shear_image)
        shear_button.grid(row=1, column=3, padx=5, pady=5)

        # Translate controls
        translate_label = ctk.CTkLabel(control_frame, text="Przesunięcie (piksele)")
        translate_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.translate_slider = ctk.CTkSlider(control_frame, from_=-50, to=50, number_of_steps=100, command=self.update_translate_label)
        self.translate_slider.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.translate_value_label = ctk.CTkLabel(control_frame, text="0 px")
        self.translate_value_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        translate_button = ctk.CTkButton(control_frame, text="Dodaj Przesunięcie", command=self.translate_image)
        translate_button.grid(row=3, column=3, padx=5, pady=5)

        # Darken controls
        darken_label = ctk.CTkLabel(control_frame, text="Ściemnianie (procent)")
        darken_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.darken_slider = ctk.CTkSlider(control_frame, from_=0.8, to=1.2, number_of_steps=40, command=self.update_darken_label)
        self.darken_slider.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.darken_value_label = ctk.CTkLabel(control_frame, text="100%")
        self.darken_value_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        darken_button = ctk.CTkButton(control_frame, text="Dodaj Ściemnianie", command=self.darken_image)
        darken_button.grid(row=4, column=3, padx=5, pady=5)

        # Noise controls
        noise_label = ctk.CTkLabel(control_frame, text="Szum (procent)")
        noise_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.noise_slider = ctk.CTkSlider(control_frame, from_=0, to=50, number_of_steps=50, command=self.update_noise_label)
        self.noise_slider.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        self.noise_value_label = ctk.CTkLabel(control_frame, text="0%")
        self.noise_value_label.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        noise_button = ctk.CTkButton(control_frame, text="Dodaj Szum", command=self.add_noise)
        noise_button.grid(row=5, column=3, padx=5, pady=5)

        # Undo, Save, and Cancel buttons
        undo_button = ctk.CTkButton(control_frame, text="Cofnij Zmianę", command=self.undo_last_change)
        undo_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2, sticky="ew")

        save_button = ctk.CTkButton(control_frame, text="Zapisz Zmiany", command=lambda: self.save_image(upload_window))
        save_button.grid(row=6, column=2, padx=5, pady=5, columnspan=2, sticky="ew")

        cancel_button = ctk.CTkButton(control_frame, text="Anuluj", command=upload_window.destroy)
        cancel_button.grid(row=7, column=0, padx=5, pady=5, columnspan=4, sticky="ew")

        # Configure grid layout
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)
    

    def capture_canvas_image(self):
        ps = self.canvas.postscript(colormode='color', width=self.canvas.winfo_width(),
        height=self.canvas.winfo_height())
        self.uploaded_image = Image.open(io.BytesIO(ps.encode('utf-8')))
        self.uploaded_image = self.uploaded_image.resize((self.canvas.winfo_width() * 1,
        self.canvas.winfo_height() * 1), Image.LANCZOS)

    def rotate_image(self):
        if hasattr(self, 'uploaded_image'):
            img_array = np.array(self.uploaded_image)
            angle = self.rotate_slider.get()
            M = cv2.getRotationMatrix2D((img_array.shape[1] / 2, img_array.shape[0] / 2), angle, 1)
            rotated = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]), borderMode=cv2.BORDER_REFLECT)
            self.uploaded_image = Image.fromarray(rotated)
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            self.history.append(self.uploaded_image.copy())

    def shear_image(self):
        if hasattr(self, 'uploaded_image'):
            img_array = np.array(self.uploaded_image)
            shear_factor = self.shear_slider.get()
            M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            sheared = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]), borderMode=cv2.BORDER_REFLECT)
            self.uploaded_image = Image.fromarray(sheared)
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            self.history.append(self.uploaded_image.copy())

    def translate_image(self):
        if hasattr(self, 'uploaded_image'):
            img_array = np.array(self.uploaded_image)
            translate_x = self.translate_slider.get()
            translate_y = self.translate_slider.get()
            M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            translated = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]), borderMode=cv2.BORDER_REFLECT)
            self.uploaded_image = Image.fromarray(translated)
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            self.history.append(self.uploaded_image.copy())

    def darken_image(self):
        if hasattr(self, 'uploaded_image'):
            img_array = np.array(self.uploaded_image)
            alpha = self.darken_slider.get()
            darkened = cv2.convertScaleAbs(img_array, alpha=alpha, beta=0)
            self.uploaded_image = Image.fromarray(darkened)
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            self.history.append(self.uploaded_image.copy())

    def add_noise(self):
        if hasattr(self, 'uploaded_image'):
            img_array = np.array(self.uploaded_image)
            noise_level = self.noise_slider.get()
            noise = np.random.normal(0, noise_level, img_array.shape).astype(np.uint8)
            noisy_img = cv2.add(img_array, noise)
            self.uploaded_image = Image.fromarray(noisy_img)
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            self.history.append(self.uploaded_image.copy())

    def undo_last_change(self):
        if len(self.history) > 1:
            self.history.pop()
            self.uploaded_image = self.history[-1]
            self.uploaded_image_tk = ImageTk.PhotoImage(self.uploaded_image)
            self.uploaded_image_label.configure(image=self.uploaded_image_tk)
            
    def update_rotate_label(self, value):
        self.rotate_value_label.configure(text=f"{int(float(value))}")

    def update_scale_label(self, value):
        self.scale_value_label.configure(text=f"{float(value):.2f}")

    def update_shear_label(self, value):
        self.shear_value_label.configure(text=f"{float(value):.2f}")

    def update_darken_label(self, value):
        self.darken_value_label.configure(text=f"{float(value):.2f}")

    def update_noise_label(self, value):
        self.noise_value_label.configure(text=f"{int(float(value))}")

    def update_translate_label(self, value):
        self.translate_value_label.configure(text=f"{int(float(value))}")

    def save_image(self, window):
        # Resize the uploaded image to match the canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas_image = self.uploaded_image.resize((canvas_width, canvas_height), Image.LANCZOS)
        
        # Save the resized image
        self.canvas_image.save("canvas_image.png")
        
        # Clear the canvas
        self.canvas.delete("all")
        
        # Update the canvas with the new image
        self.canvas_image_tk = ImageTk.PhotoImage(self.canvas_image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.canvas_image_tk)
        
        # Update self.image with the new canvas image
        self.image = self.canvas_image.convert('L')
        
        # Close the upload window
        window.destroy()
        
if __name__ == "__main__":
    root = ctk.CTk()
    app = KatakanaRecognizerApp(root)
    root.mainloop()