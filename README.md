# SignEase - Aplikacja do rozpoznawania języka migowego

Projekt realizowany w ramach zajęć Inżynierii oprogramowania w semestrze zimowym roku akademickiego 2024/2025.

**Autorzy:**
- Patrycja Sapko
- Izabela Jesionowska

## 1. Charakterystyka oprogramowania

**Nazwa skrócona:** SE  
**Nazwa pełna:** SignEase – połączenie słów "sign" (gest) i "ease" (łatwość), co sugeruje łatwość w nauce i komunikacji w języku migowym.

### Krótki opis i cele:
Aplikacja służy do rozpoznawania znaków języka migowego w czasie rzeczywistym przy użyciu kamery internetowej. System wykorzystuje techniki uczenia maszynowego i przetwarzania obrazu do identyfikacji gestów wykonywanych przez użytkownika.

Główne cele:
- Umożliwienie rozpoznawania znaków języka migowego w czasie rzeczywistym
- Stworzenie bazy danych gestów języka migowego
- Zapewnienie możliwości trenowania modelu na podstawie własnych przykładów
- Ułatwienie komunikacji z osobami posługującymi się językiem migowym
- Wsparcie procesu nauki języka migowego poprzez natychmiastową informację zwrotną

## 2. Prawa autorskie

### Wykorzystane biblioteki i licencje:

| Biblioteka  | Licencja      |
|------------|---------------|
| OpenCV     | Apache 2.0    |
| MediaPipe  | Apache 2.0    |
| PyQt5      | GPL v3        |
| NumPy      | BSD          |
| scikit-learn| BSD 3-Clause |
| pygame     | LGPL         |

*Zgodnie z warunkami licencji GPL v3 (ze względu na wykorzystanie PyQt5), cały projekt musi być udostępniany na tej samej licencji.*

## 3. Specyfikacja wymagań

### Wymagania funkcjonalne:

#### 1. Zarządzanie danymi treningowymi:
- Możliwość dodawania nowych znaków do bazy danych
- Zbieranie obrazów treningowych poprzez kamerę
- Etykietowanie zebranych danych

#### 2. Trening modelu:
- Przetwarzanie zebranych danych obrazowych
- Ekstrakcja cech charakterystycznych gestów
- Trenowanie klasyfikatora Random Forest
- Zapisywanie wytrenowanego modelu

#### 3. Rozpoznawanie znaków:
- Przechwytywanie obrazu z kamery w czasie rzeczywistym
- Detekcja i śledzenie dłoni
- Klasyfikacja wykrytych gestów
- Wyświetlanie rozpoznanych znaków na ekranie
- Odtwarzanie dźwięku odpowiadającego rozpoznanemu znakowi

#### 4. Interfejs użytkownika:
- Przycisk do rozpoczęcia rozpoznawania
- Funkcja dodawania nowych znaków
- Opcja trenowania modelu
- Wskaźnik postępu podczas trenowania
- Wyświetlanie komunikatów o błędach i statusie operacji

### Szczegółowe wymagania funkcjonalne:

| ID    | Nazwa | Opis | Priorytet |
|-------|-------|------|-----------|
| WF.1  | Zarządzanie danymi treningowymi | System umożliwia zbieranie i zarządzanie danymi treningowymi poprzez nagrywanie i etykietowanie gestów języka migowego. | 1 |
| WF.1.1| Dodawanie nowego znaku | System umożliwia dodanie nowego znaku do bazy danych poprzez nagranie serii gestów przy użyciu kamery. | 1 |
| WF.1.2| Zbieranie obrazów treningowych | System automatycznie zapisuje serię zdjęć podczas nagrywania nowego znaku, tworząc bazę danych treningowych. | 1 |
| WF.2  | Trening modelu | System przetwarza zebrane dane i trenuje model uczenia maszynowego do rozpoznawania gestów. | 1 |
| WF.3  | Rozpoznawanie znaków w czasie rzeczywistym | System w czasie rzeczywistym analizuje obraz z kamery i rozpoznaje wykonywane gesty języka migowego. | 1 |
| WF.3.1| Informacja dźwiękowa | System odtwarza dźwiękowy komunikat odpowiadający rozpoznanemu znakowi. | 2 |

### Wymagania pozafunkcjonalne:

#### 1. Wydajność:
- Rozpoznawanie znaków w czasie rzeczywistym (min. 15 FPS)
- Czas odpowiedzi interfejsu użytkownika poniżej 100ms
- Obsługa minimum 100 różnych znaków

#### 2. Niezawodność:
- Logowanie błędów i zdarzeń do pliku
- Obsługa wyjątków i błędów
- Zabezpieczenie przed utratą danych podczas treningu

#### 3. Użyteczność:
- Intuicyjny interfejs użytkownika
- Czytelne komunikaty o błędach
- Wizualna informacja zwrotna podczas rozpoznawania

#### 4. Kompatybilność:
- Wsparcie dla systemów Windows
- Kompatybilność z większością kamer internetowych

#### 5. Bezpieczeństwo:
- Lokalne przetwarzanie danych
- Bezpieczne zapisywanie danych treningowych
- Ochrona przed nieautoryzowanym dostępem do modelu

### Szczegółowe wymagania pozafunkcjonalne:

| ID    | Nazwa | Opis | Priorytet |
|-------|-------|------|-----------|
| WN.1  | Wydajność rozpoznawania | System musi działać w czasie rzeczywistym z minimalną częstotliwością 15 klatek na sekundę. | 1 |
| WN.2  | Dokładność rozpoznawania | System musi osiągać dokładność rozpoznawania gestów na poziomie minimum 85%. | 1 |
| WN.3  | Niezawodność | System musi działać stabilnie przez minimum 8 godzin ciągłej pracy. | 2 |
| WN.4  | Logowanie zdarzeń | System musi zapisywać wszystkie istotne zdarzenia i błędy w pliku logów. | 2 |

## 4. Architektura oprogramowania

### Architektura rozwoju

#### Python i Podstawowe Biblioteki:
- **Python**
  - Przeznaczenie: Język programowania wykorzystywany jako podstawa aplikacji
  - Wersja: 3.x

#### Biblioteki do Przetwarzania Obrazu i ML:
- **OpenCV (cv2)**
  - Przeznaczenie: Przechwytywanie i przetwarzanie obrazu z kamery
  - Wersja: 4.x
- **MediaPipe**
  - Przeznaczenie: Detekcja i śledzenie punktów charakterystycznych dłoni
  - Wersja: Latest stable
- **NumPy**
  - Przeznaczenie: Operacje na macierzach i tablicach wielowymiarowych
  - Wersja: Latest stable
- **scikit-learn**
  - Przeznaczenie: Implementacja algorytmów uczenia maszynowego (Random Forest Classifier)
  - Wersja: Latest stable

#### Interfejs Użytkownika:
- **PyQt5**
  - Przeznaczenie: Framework GUI
  - Wersja: 5.x

#### System Dźwięku:
- **Pygame**
  - Przeznaczenie: Odtwarzanie dźwięków dla rozpoznanych znaków
  - Wersja: Latest stable

#### Przechowywanie Danych:
- **pickle**
  - Przeznaczenie: Serializacja modelu ML i danych treningowych
  - Wersja: Wbudowany w Python

### Architektura uruchomieniowa

#### Wymagania Systemowe:
- System operacyjny: Windows/Linux/MacOS
- Minimalna pamięć RAM: 4GB
- Procesor: Multi-core CPU
- Kamera internetowa: Wymagana

#### Struktura katalogów:
```
.
├── data/               # dane treningowe
├── sounds/            # pliki dźwiękowe
├── models/            # zapisane modele
├── logs/              # pliki logów
├── model.p           # wytrenowany model
├── data.pickle       # przetworzone dane treningowe
└── migowy.py         # główny plik aplikacji
```

### Przepływ danych:

#### Zbieranie danych:
```
Kamera -> Przechwytywanie obrazu -> Detekcja dłoni -> Ekstrakcja cech -> Zapis do bazy danych
```

#### Trening modelu:
```
Baza danych -> Preprocessing -> Trening klasyfikatora -> Zapisany model
```

#### Rozpoznawanie:
```
Kamera -> Detekcja dłoni -> Ekstrakcja cech -> Klasyfikacja -> Wyświetlenie wyniku -> Odtworzenie
```

### Proces wdrożenia:
1. Instalacja Pythona i wymaganych bibliotek
2. Konfiguracja środowiska:
   - Utworzenie katalogów dla danych treningowych
   - Konfiguracja parametrów kamery
   - Ustawienie ścieżek do plików dźwiękowych
3. Uruchomienie aplikacji:
   ```bash
   python migowy.py
   ```

## 5. Testy

### Scenariusze testowe:

| ID | Nazwa scenariusza | Kroki | Oczekiwany rezultat |
|----|------------------|-------|-------------------|
| INIT-001 | Sprawdzenie poprawności uruchomienia aplikacji | 1. Uruchom aplikację<br>2. Sprawdź czy interfejs GUI się wyświetla<br>3. Zweryfikuj obecność wszystkich przycisków | Aplikacja uruchamia się bez błędów, wszystkie elementy GUI są widoczne |
| ADD-001 | Sprawdzenie funkcjonalności dodawania nowego znaku do bazy danych | 1. Kliknij „Dodaj znak"<br>2. Wprowadź etykietę znaku<br>3. Wykonaj serię gestów przed kamerą | 100 zdjęć zostaje zapisanych w odpowiednim katalogu |
| TRAIN-001 | Weryfikacja procesu trenowania modelu | 1. Kliknij "Analizuj i trenuj model"<br>2. Poczekaj na zakończenie procesu<br>3. Sprawdź komunikat o wynikach | Model zostaje wytrenowany, wyświetlone zostają statystyki sukcesu |
| RECOG-001 | Sprawdzenie funkcjonalności rozpoznawania znaków | 1. Kliknij "Rozpoznawanie"<br>2. Wykonaj znany modelowi gest<br>3. Sprawdź czy znak został rozpoznany | Aplikacja poprawnie rozpoznaje gest i wyświetla etykietę |

### Sprawozdanie z wykonania testów:

#### Test INIT-001
- **Status:** PASS
- **Uwagi:** Interfejs uruchamia się poprawnie na wszystkich testowanych systemach operacyjnych
- **Znalezione problemy:** Brak

#### Test ADD-001
- **Status:** PASS
- **Uwagi:**
  - Proces przebiega zgodnie z oczekiwaniami
  - Zdjęcia są zapisywane w odpowiedniej strukturze katalogów
- **Znalezione problemy:**
  - Czasami występują problemy z detekcją dłoni w słabym oświetleniu

#### Test TRAIN-001
- **Status:** PASS
- **Uwagi:**
  - Proces trenowania działa poprawnie
  - Wyświetlany jest pasek postępu
  - Generowany jest raport z procesu trenowania
- **Znalezione problemy:**
  - Przy bardzo dużej liczbie zdjęć proces może być czasochłonny

#### Test RECOG-001
- **Status:** PASS
- **Uwagi:**
  - Rozpoznawanie działa w czasie rzeczywistym
  - Etykiety są wyświetlane na obrazie z kamery
  - Dźwięki są odtwarzane poprawnie
- **Znalezione problemy:**
  - Czasami występują fałszywe rozpoznania przy szybkich ruchach
  - Może wystąpić opóźnienie w odtwarzaniu dźwięku

### Ogólne wnioski z testów:
1. Aplikacja działa stabilnie i spełnia podstawowe wymagania funkcjonalne
2. Główne problemy związane są z warunkami oświetleniowymi i szybkością ruchów
3. Zalecane jest przeprowadzenie dodatkowych testów w różnych warunkach oświetleniowych

