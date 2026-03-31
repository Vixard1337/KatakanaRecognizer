import tkinter as tk
from tkinter import filedialog, messagebox

def recognize_characters():
    # Tutaj dodaj logikę rozpoznawania znaków
    messagebox.showinfo("Info", "Rozpoznawanie znaków...")

def main():
    root = tk.Tk()
    root.title("Katakana Recognition")

    label = tk.Label(root, text="Wybierz plik do rozpoznania:")
    label.pack(pady=10)

    button = tk.Button(root, text="Wybierz plik", command=recognize_characters)
    button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()