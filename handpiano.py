"""
El Piyanosu - iki el, başparmağa dokunan parmak ile 16 nota.

Sağ el : notayı SEÇER ve ÇALAR (başparmak + işaret / orta / yüzük / serçe)
Sol el : oktav bankasını seçer, basılı tutulur (aynı 4 dokunma)

4 banka x 4 parmak = 16 nota -> Do majör pentatonik, C3 - C6.
Pentatonik seçildi: yanlış hücreye bassan da uyumsuz ses çıkmaz.

Sol el hiçbir şey yapmıyorsa orta banka kullanılır, yani tek elle de çalınabilir.
Notalar birbirini kesmez, üst üste binerek söner.

Çıkmak için pencere açıkken 'q' tuşuna bas.

Kurulum:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
"""
import os
import time
import urllib.request

import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Ayarlar ----------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

SAMPLE_RATE = 44100
NOTE_SECONDS = 1.0      # bir notanın sönme süresi
STABLE_FRAMES = 3       # nota çalmadan önce el durumunun kaç kare sabit kalması gerektiği
PINCH_RATIO = 0.45      # dokunma eşiği: (başparmak-parmak) / (el ölçeği). Büyütürsen daha kolay tetiklenir.
DEFAULT_BANK = 1        # sol el boşken kullanılan banka (0-3)

THUMB_TIP = 4
FINGER_TIPS = [8, 12, 16, 20]                       # işaret, orta, yüzük, serçe
FINGER_LABELS = ["Isaret", "Orta", "Yuzuk", "Serce"]  # ekranda ASCII şart (OpenCV yazı tipi)

# satır = sol el parmağı (banka), sütun = sağ el parmağı (nota)
NOTE_GRID = [
    ["C3", "D3", "E3", "G3"],
    ["A3", "C4", "D4", "E4"],
    ["G4", "A4", "C5", "D5"],
    ["E5", "G5", "A5", "C6"],
]

# MediaPipe el noktaları arasındaki bağlantılar (çizim için)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),            # işaret
    (5, 9), (9, 10), (10, 11), (11, 12),       # orta
    (9, 13), (13, 14), (14, 15), (15, 16),     # yüzük
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # serçe + avuç
]


# ---------------- Yardımcılar ----------------
def ensure_model():
    """El takip modelini ilk çalıştırmada indirir."""
    if not os.path.exists(MODEL_PATH):
        print("El modeli indiriliyor (bir kereye mahsus)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def note_freq(name):
    """'C4' -> Hz. Eşit tamperaman, A4 = 440 Hz."""
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    semis = steps[name[0]] + 12 * (int(name[1:]) - 4) - 9   # A4'ten kaç yarım ses uzakta
    return 440.0 * (2 ** (semis / 12))


def make_tone(freq, duration=NOTE_SECONDS):
    """Basit piyano benzeri ton üretir: birkaç harmonik + sönümlenen zarf."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.25 * np.sin(2 * np.pi * 3 * freq * t))

    envelope = np.exp(-3 * t)                       # zamanla sönme
    attack = int(0.01 * SAMPLE_RATE)
    envelope[:attack] *= np.linspace(0, 1, attack)  # tık sesini önlemek için yumuşak giriş
    wave *= envelope
    # 16 nota üst üste binebildiği için tepe seviyesi düşük tutuluyor (kırpılma olmasın)
    wave = wave / np.max(np.abs(wave)) * 0.4

    samples = (wave * 32767).astype(np.int16)
    if pygame.mixer.get_init()[2] == 2:             # stereo ise iki kanala kopyala
        samples = np.column_stack([samples, samples])
    return pygame.sndarray.make_sound(np.ascontiguousarray(samples))


def px_dist(a, b, w, h):
    """Piksel uzayında mesafe. Normalize koordinatları doğrudan kullanmak
    en-boy oranı yüzünden x'i ezer, bu yüzden w/h ile ölçekliyoruz."""
    return (((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2) ** 0.5


def pinch_finger(lm, w, h):
    """Başparmağa dokunan parmağın indeksini (0-3) döndürür, dokunan yoksa None.

    Eşik el ölçeğine (bilek -> orta parmak kökü) göre normalize edildiği için
    elin kameraya uzaklığı sonucu etkilemez.
    """
    ref = px_dist(lm[0], lm[9], w, h)
    if ref < 1e-6:
        return None
    best, best_ratio = None, PINCH_RATIO
    for i, tip in enumerate(FINGER_TIPS):
        ratio = px_dist(lm[THUMB_TIP], lm[tip], w, h) / ref
        if ratio < best_ratio:                      # en yakın parmak kazanır
            best, best_ratio = i, ratio
    return best


def user_hand(category_name):
    """Görüntüyü aynaladığımız için MediaPipe'ın el etiketi ters gelir; düzeltiyoruz."""
    return "Right" if category_name == "Left" else "Left"


def draw_hand(frame, lm, color):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_grid(frame, bank, finger):
    """4x4 nota tablosunu çizer; aktif banka ve çalınacak nota vurgulanır."""
    x0, y0, cw, ch = 20, 96, 92, 38
    for c in range(4):
        cv2.putText(frame, FINGER_LABELS[c][:5], (x0 + c * cw + 4, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    for r in range(4):
        for c in range(4):
            x, y = x0 + c * cw, y0 + r * ch
            active = (r == bank and c == finger)
            fill = (0, 220, 255) if active else ((70, 70, 70) if r == bank else (40, 40, 40))
            cv2.rectangle(frame, (x, y), (x + cw - 4, y + ch - 4), fill, -1)
            cv2.rectangle(frame, (x, y), (x + cw - 4, y + ch - 4), (110, 110, 110), 1)
            txt_color = (0, 0, 0) if active else (235, 235, 235)
            cv2.putText(frame, NOTE_GRID[r][c], (x + 14, y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)
        if r == bank:                               # aktif bankayı solda işaretle
            cv2.putText(frame, ">", (x0 - 16, y0 + r * ch + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)


# ---------------- Ana döngü ----------------
def main():
    ensure_model()

    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(16)               # notalar üst üste binebilsin
    sounds = {name: make_tone(note_freq(name))
              for row in NOTE_GRID for name in row}

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı. Sistem Ayarları > Gizlilik > Kamera'dan "
              "VS Code / Terminal'e izin verdiğinden emin ol.")
        return

    last_played = None                              # en son çalınan (banka, parmak)
    candidate, candidate_frames = None, 0
    start, last_ts = time.time(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                  # ayna görüntüsü, daha doğal hissettirir
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - start) * 1000), last_ts + 1)  # artan zaman damgası şart
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        left_pinch, right_pinch = None, None
        for lm, handed in zip(result.hand_landmarks, result.handedness):
            side = user_hand(handed[0].category_name)
            if side == "Left":
                left_pinch = pinch_finger(lm, w, h)
                draw_hand(frame, lm, (255, 120, 0))     # sol el: banka seçici
            else:
                right_pinch = pinch_finger(lm, w, h)
                draw_hand(frame, lm, (0, 0, 255))       # sağ el: notayı çalar

        bank = left_pinch if left_pinch is not None else DEFAULT_BANK

        # Titremeyi önle: aynı (banka, parmak) ikilisi birkaç kare sürerse kabul et
        state = (bank, right_pinch)
        if state == candidate:
            candidate_frames += 1
        else:
            candidate, candidate_frames = state, 1

        # Sadece durum DEĞİŞTİĞİNDE ve sağ el bir parmağa dokunuyorken nota çal
        if candidate_frames == STABLE_FRAMES and candidate != last_played:
            if right_pinch is not None:
                sounds[NOTE_GRID[bank][right_pinch]].play()   # öncekini kesmiyoruz
            last_played = candidate

        draw_grid(frame, bank, right_pinch)

        note = NOTE_GRID[bank][right_pinch] if right_pinch is not None else "-"
        hud = "Banka: {}{}   Nota: {}".format(
            FINGER_LABELS[bank], "" if left_pinch is not None else " (varsayilan)", note)
        cv2.putText(frame, hud, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("El Piyanosu", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
