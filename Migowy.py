import os
import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QDialog, QVBoxLayout, QProgressBar, QLabel
import threading
import sys
import pygame  # Importowanie pygame do odtwarzania dźwięku

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

#Tworzy okienko pokazujące pasek postępu 
class ProgressDialog(QDialog):
    """Okno dialogowe paska postępu."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proszę czekać")
        self.setGeometry(200, 200, 300, 100)

        self.layout = QVBoxLayout()
        self.label = QLabel("Trwa tworzenie zbioru danych i trenowanie modelu...")
        self.layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)

        self.setLayout(self.layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

#Główne okno aplikacji zawierające trzy przyciski
class SignLanguageApp(QtWidgets.QWidget):
    update_progress = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.progress_dialog = None
        self.update_progress.connect(self.update_progress_in_dialog)

    def initUI(self):
        self.setWindowTitle("Rozpoznawanie Języka Migowego")
        self.setGeometry(100, 100, 500, 400)
        self.setStyleSheet("background-color: #F3F3F3;")
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addStretch(1)
        additional_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_run_inference = QtWidgets.QPushButton("Rozpoznawanie")
        self.btn_run_inference.clicked.connect(self.on_run_inference)
        self.style_main_button(self.btn_run_inference)
        main_layout.addWidget(self.btn_run_inference, alignment=QtCore.Qt.AlignCenter)
        main_layout.addStretch(1)
        additional_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_collect_images = QtWidgets.QPushButton("Dodaj znak")
        self.btn_collect_images.clicked.connect(self.on_collect_images)
        self.style_button(self.btn_collect_images)
        additional_buttons_layout.addWidget(self.btn_collect_images)
        self.btn_analyze_and_train = QtWidgets.QPushButton("Analizuj i trenuj model")
        self.btn_analyze_and_train.clicked.connect(self.on_analyze_and_train)
        self.style_button(self.btn_analyze_and_train)
        additional_buttons_layout.addWidget(self.btn_analyze_and_train)
        main_layout.addLayout(additional_buttons_layout)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def style_main_button(self, button):
        button.setFixedHeight(60)
        button.setFont(QtGui.QFont("Segoe UI Variable", 14, QtGui.QFont.Bold))
        button.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
        """)
        self.add_shadow(button)

    def style_button(self, button):
        button.setFixedHeight(50)
        button.setFont(QtGui.QFont("Segoe UI Variable", 12))
        button.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
        """)
        self.add_shadow(button)

    def add_shadow(self, button):
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 5)
        shadow.setColor(QtGui.QColor(0, 0, 0, 80))
        button.setGraphicsEffect(shadow)

    def update_progress_in_dialog(self, value):
        if self.progress_dialog:
            self.progress_dialog.update_progress(value)

    def on_collect_images(self):
        sign_label, ok = QInputDialog.getText(self, 'Dodaj znak', 'Podaj etykietę znaku:')
        if ok and sign_label:
            threading.Thread(target=collect_images, args=("data", sign_label, 100, self)).start()

    def show_collect_success(self):
        QMessageBox.information(self, "Info", "Znak dodany!")

    def on_analyze_and_train(self):
        self.progress_dialog = ProgressDialog()
        self.progress_dialog.show()

        def analyze_and_train():
            print("Rozpoczęto tworzenie zbioru danych...")
            success_rate_images = create_dataset(window_ref=self)
            print("Tworzenie zbioru danych zakończone. Rozpoczęto trenowanie modelu...")
            training_success_rate = train_classifier(window_ref=self)
            print("Trenowanie modelu zakończone.")

            message = (f"Tworzenie bazy danych i trening modeli został przeprowadzony!\n"
                       f"Procent poprawnie przetwarzonych obrazów: {success_rate_images:.2f}%\n"
                       f"Procent udanych iteracji treningu: {training_success_rate:.2f}%")
            QtCore.QMetaObject.invokeMethod(self, "show_train_complete", QtCore.Qt.QueuedConnection, 
                                            QtCore.Q_ARG(str, message))

        threading.Thread(target=analyze_and_train).start()

    @QtCore.pyqtSlot(str)
    def show_train_complete(self, message):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.information(self, "Info", message)

    def on_run_inference(self):
        threading.Thread(target=run_inference, args=(self,)).start()

    @QtCore.pyqtSlot()
    def show_train_model_error(self):
        QMessageBox.critical(self, "Błąd", "Model nie został znaleziony. Proszę najpierw przeprowadzić trening modelu.")

#Funkcja uruchamiana po kliknięciu "Dodaj znak"
def collect_images(data_dir, label, num_images, window_ref):
    data_dir = os.path.join(BASE_DIR, data_dir, label)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        cv2.putText(frame, 'Nacisnij Enter aby rozpoczac', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        img_path = os.path.join(data_dir, f'{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1
        print(f'Captured image {count}/{num_images}')
        cv2.putText(frame, f'Capturing images: {count}/{num_images}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    QtCore.QTimer.singleShot(0, window_ref.show_collect_success)

#Funkcja przetwarzająca zebrane zdjęcia
def create_dataset(data_dir='data', output_file='data.pickle', window_ref=None):
    data_dir = os.path.join(BASE_DIR, data_dir)
    output_file = os.path.join(BASE_DIR, output_file)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    data = []
    labels = []
    total_images = sum(len(files) for _, _, files in os.walk(data_dir))
    processed_images = 0
    successful_images = 0

    for dir_ in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, dir_)
        if not os.path.isdir(class_dir):
            continue
        print(f"Przetwarzanie klasy: {dir_}")
        for img_path in os.listdir(class_dir):
            data_aux = []
            x_ = []
            y_ = []
            img = cv2.imread(os.path.join(class_dir, img_path))
            if img is None:
                print(f"Error: Could not read image {img_path}.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
                    successful_images += 1
                    print(f"Przetworzono obraz: {img_path}")
            else:
                print(f"No hand landmarks found in image: {img_path}")
            processed_images += 1
            progress = int((processed_images / total_images) * 50)
            window_ref.update_progress.emit(progress)

    success_rate_images = (successful_images / total_images) * 100

    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Zbiór danych zapisano do {output_file}")

    return success_rate_images

#Uczy model rozpoznawać znaki
def train_classifier(data_file='data.pickle', model_file='model.p', window_ref=None):
    data_file = os.path.join(BASE_DIR, data_file)
    model_file = os.path.join(BASE_DIR, model_file)
    try:
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
        print("Wczytano zbiór danych do trenowania.")
    except FileNotFoundError:
        print(f"Error: Dataset file {data_file} not found.")
        return

    data = np.asarray(data_dict['data'], dtype=object)
    labels = np.asarray(data_dict['labels'])
    filtered_data = []
    filtered_labels = []
    label_counts = Counter(labels)

    for i in range(len(labels)):
        if label_counts[labels[i]] >= 2:
            filtered_data.append(data[i])
            filtered_labels.append(labels[i])

    x_train, x_test, y_train, y_test = train_test_split(
        filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    print("Trenowanie modelu...")
    successful_training_steps = 0
    for i in range(1, 101):
        model.fit(x_train, y_train)
        successful_training_steps += 1
        progress = 50 + int(i / 100 * 50)
        window_ref.update_progress.emit(progress)

    training_success_rate = (successful_training_steps / 100) * 100

    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'labels': list(set(filtered_labels))}, f)
    print(f"Model zapisano do {model_file}")

    return training_success_rate

#Odtwarza dźwięk odpowiadający rozpoznanemu znakowi
def play_sound_for_sign(sign_label):
    sound_file = os.path.join(BASE_DIR, f"sounds/{sign_label}.mp3")
    if os.path.exists(sound_file):
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    else:
        print(f"Dźwięk dla znaku '{sign_label}' nie został znaleziony.")

#Główna funkcja rozpoznająca znaki. Wczytuje wytrenowany model, włącza kamerę i wykrywa dłoń w czasie rzeczywistym.
def run_inference(window_ref, model_file='model.p'):
    model_file = os.path.join(BASE_DIR, model_file)
    if not os.path.exists(model_file):
        QtCore.QMetaObject.invokeMethod(window_ref, "show_train_model_error", QtCore.Qt.QueuedConnection)
        return

    model_dict = pickle.load(open(model_file, 'rb'))
    model = model_dict['model']
    labels_dict = {label: label for label in model_dict['labels']}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    stop_flag = False
    last_seen_character = None
    last_seen_time = 0
    play_delay = 2

    def close_window():
        nonlocal stop_flag
        stop_flag = True

    cv2.namedWindow('frame')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while not stop_flag:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                x_ = [hand_landmarks.landmark[i].x for i in range(21)]
                y_ = [hand_landmarks.landmark[i].y for i in range(21)]
                for i in range(21):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))
            if len(data_aux) > 42:
                data_aux = data_aux[:42]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[prediction[0]]

            if predicted_character == last_seen_character:
                elapsed_time = time.time() - last_seen_time
                if elapsed_time >= play_delay:
                    play_sound_for_sign(predicted_character)
                    last_seen_character = None
            else:
                last_seen_character = predicted_character
                last_seen_time = time.time()

            if predicted_character in labels_dict.values():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == 27:
            close_window()
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
            break

    cap.release()
    cv2.destroyWindow('frame')
    cv2.destroyAllWindows()

#Uruchamia aplikację
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())


