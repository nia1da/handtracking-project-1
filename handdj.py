"""
El DJ - çalan bir parçayı iki elle gerçek zamanlı manipüle eder.

Her el bağımsız çalışır:
    hangi parmak başparmağa dokunuyorsa  -> o efekt seçilir
    elin yüksekliği                      -> efektin miktarı (aşağı = 0, yukarı = 1)

    SOL EL                      SAĞ EL
    işaret : Lowpass            işaret : Bas
    orta   : Highpass           orta   : Hız (varispeed)
    yüzük  : Delay              yüzük  : Tremolo
    serçe  : Bitcrush           serçe  : Ses

İki el aynı anda = 4 x 4 = 16 efekt çifti.

Dokunmayı bıraktığında ve parmak değiştirdiğinde efekt SON DEĞERİNDE kalır,
nötre dönmez. O parmağa yeniden dokunduğunda efekt elinin yüksekliğini alır.
Yakalarken değerin yerinde kalmasını istersen RELATIVE_TAKEOVER = True yap.

Hız varispeed: pitch ve tempo birlikte değişir (pikap gibi), 0.5x - 2.0x.

Tuşlar: q = çık, boşluk = duraklat/devam, r = tüm efektleri nötre sıfırla

Kullanım:
    venv/bin/python handdj.py [parca.mp3]
Dosya verilmezse klasördeki ilk ses dosyasını arar, onu da bulamazsa
test amaçlı sentetik bir davul-bas loop'u üretir.
"""
import os
import sys
import glob
import time
import struct
import shutil
import tempfile
import subprocess

import cv2
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Ayarlar ----------------
MODEL_PATH = "hand_landmarker.task"
SR = 48000              # çıkış aygıtının doğal hızı
BLOCK = 512             # ~10.7 ms gecikme
SMOOTH = 0.18           # parametre yumuşatma (zipper gürültüsünü önler)
PINCH_RATIO = 0.45      # dokunma eşiği, el ölçeğine göre normalize
Y_TOP, Y_BOTTOM = 0.15, 0.85   # el yüksekliğinin kullanılabilir bandı
# False (varsayılan): efekt doğrudan elin yüksekliğini alır. Yakalama anında
#   değer elin bulunduğu yere geçer; SMOOTH yüzünden bu ~150 ms'lik yumuşak bir
#   süzülme olarak duyulur, tık sesi değil. Tam aralık her zaman elinin altında.
# True: yakalarken değer yerinde kalır, sonra elinin hareketi kadar kayar.
#   Zıplama olmaz ama eli yukarıda yakalarsan yukarı az yer kalır; tam aralık için
#   bırak-indir-yeniden yakala gerekir (fare tırmıklaması gibi).
RELATIVE_TAKEOVER = False

# Efekt derinliği çarpanı. 1.0 = ilk sürümdeki aralıklar, 1.5 = %150.
# Her efekte uygulanıyor ama hepsi 1.5 katı KALDIRAMIYOR; fiziksel sınıra
# dayananlar (tremolo derinliği, bitcrush bit sayısı) aşağıda kırpılıyor.
FX_SCALE = 1.5

# Filtre uçları oktav cinsinden ölçekleniyor (frekansta çarpmak anlamsız olur).
# Taban/tavan değerleri sesi tamamen yok etmemek için sınırlı.
_LP_HI, _LP_REF = 12000.0, 150.0
LP_LO = max(30.0, _LP_HI / 2.0 ** (np.log2(_LP_HI / _LP_REF) * FX_SCALE))
_HP_LO, _HP_REF = 20.0, 2000.0
HP_HI = min(8000.0, _HP_LO * 2.0 ** (np.log2(_HP_REF / _HP_LO) * FX_SCALE))
BASS_DB_MAX = 12.0 * FX_SCALE
CRUSH_BITS_MIN = max(1.0, 16.0 - 13.0 * FX_SCALE)
# Bitcrush'ın kuantalama ızgarası mutlak seviyeye göre çalışır: sinyal zayıfsa
# düşük bit sayılarında her örnek sıfıra yuvarlanır ve ses tamamen susar
# (crunch değil, sessizlik). Zincirdeki filtreler sinyali 0.07'ye kadar
# düşürebildiği için sabit ön kazanç yetmiyor; seviyeyi takip edip ızgarayı
# sinyale göre ölçekliyoruz.
CRUSH_TARGET = 0.9       # kuantalama öncesi hedeflenen tepe
CRUSH_FLOOR = 0.02       # sessizlikte kazancın patlamaması için taban
CRUSH_RELEASE = 0.95     # blok başına düşüş (hızlı yukarı, yavaş aşağı)
TREM_DEPTH_MAX = min(1.0, 0.9 * FX_SCALE)
DELAY_MIX_MAX = min(0.95, 0.5 * FX_SCALE)
DELAY_FB = min(0.60, 0.35 * FX_SCALE)

# Filtreler enerji siler, yani telafi olmazsa süpürme "ton değişimi" değil
# "ses kısılması" gibi duyulur. Kayıp kadar geri kazanç veriyoruz (sadece
# yükseltme yönünde) ve tavanla sınırlıyoruz, yoksa dipteki uğultu patlar.
FLT_MAKEUP_MAX = 4.0     # +12 dB tavan
FLT_MAKEUP_SMOOTH = 0.20

# Bas tek yönlü yükseltme olunca ses seviyesi fırlıyor ve telafi kazancı
# ortayı/tizi kısıyor; ikisi birbirini örtüp "bas artmadı" hissi veriyor.
# Çift yönlü (gerçek DJ EQ'su gibi): orta = düz, aşağı = kesik, yukarı = dolgun.
# Kesme tarafı çok belirgin duyulduğu için fark hissedilmemesi imkânsız.
BASS_BIPOLAR = True

LIMIT_KNEE = 0.70        # bu seviyenin üstünde yumuşak doyma (sert kırpma yerine)

THUMB_TIP = 4
FINGER_TIPS = [8, 12, 16, 20]

# (anahtar, ekran adı, nötr değer)  - sıra parmak sırasıdır: işaret, orta, yüzük, serçe
LEFT_FX = [("lowpass", "Lowpass", 0.0), ("highpass", "Highpass", 0.0),
           ("delay", "Delay", 0.0), ("crush", "Bitcrush", 0.0)]
RIGHT_FX = [("bass", "Bas", 0.5 if BASS_BIPOLAR else 0.0), ("speed", "Hiz", 0.5),
            ("trem", "Tremolo", 0.0), ("vol", "Ses", 1.0)]
ALL_FX = LEFT_FX + RIGHT_FX

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


# ---------------- Parça yükleme ----------------
def synth_loop(bars=2, bpm=120):
    """Test için sentetik davul + bas loop'u. Efektleri duymaya yeter."""
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars)
    out = np.zeros(n, dtype=np.float32)

    def add(at, buf):
        i = int(at * SR)
        k = min(len(buf), n - i)
        if k > 0:
            out[i:i + k] += buf[:k]

    def kick():
        t = np.linspace(0, 0.18, int(SR * 0.18), endpoint=False)
        f = 110 * np.exp(-28 * t) + 48          # düşen pitch
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-14 * t) * 0.9).astype(np.float32)

    def snare():
        t = np.linspace(0, 0.14, int(SR * 0.14), endpoint=False)
        rng = np.random.default_rng(7)
        return ((rng.standard_normal(len(t)) * 0.5 + np.sin(2 * np.pi * 190 * t) * 0.4)
                * np.exp(-26 * t) * 0.55).astype(np.float32)

    def hat():
        t = np.linspace(0, 0.05, int(SR * 0.05), endpoint=False)
        rng = np.random.default_rng(3)
        return (rng.standard_normal(len(t)) * np.exp(-70 * t) * 0.22).astype(np.float32)

    def bassnote(freq, dur):
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        w = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        env = np.minimum(1, np.exp(-2.5 * t) * 1.2)
        return (w * env * 0.33).astype(np.float32)

    k, s, h = kick(), snare(), hat()
    line = [55.0, 55.0, 82.41, 73.42]           # A1 A1 E2 D2
    for b in range(bars):
        b0 = b * 4 * beat
        for q in (0, 2):
            add(b0 + q * beat, k)
        add(b0 + 1.5 * beat, k)
        for q in (1, 3):
            add(b0 + q * beat, s)
        for e in range(8):
            add(b0 + e * beat * 0.5, h)
        add(b0, bassnote(line[(b * 2) % 4], beat * 1.5))
        add(b0 + 2 * beat, bassnote(line[(b * 2 + 1) % 4], beat * 2))

    out /= max(np.max(np.abs(out)), 1e-9)
    return np.column_stack([out, out]) * 0.5


def read_wav(path):
    """Küçük RIFF okuyucu -> (pcm baytları, kanal sayısı).

    Neden stdlib `wave` değil: afconvert sıkıştırılmış girdiyi (mp3/m4a/aac)
    çevirirken WAVE_FORMAT_EXTENSIBLE (fmt etiketi 65534) üretiyor ve `wave`
    bunu "unknown format: 65534" diye reddediyor. Düz WAV girdide sorun
    çıkmıyor, o yüzden bu hata sadece gerçek müzik dosyalarında görünür.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("geçerli bir WAV değil: %s" % path)
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate = struct.unpack_from("<HHI", body, 0)
            bits = struct.unpack_from("<H", body, 14)[0]
            if tag == 0xFFFE and size >= 40:        # extensible: gerçek format SubFormat GUID'inde
                tag = struct.unpack_from("<H", body, 24)[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)                # tek boyutlu chunk'lar dolgulu
    if fmt is None or pcm is None:
        raise RuntimeError("WAV içinde fmt/data chunk'ı yok")
    tag, ch, rate, bits = fmt
    if tag != 1 or bits != 16:
        raise RuntimeError("beklenmeyen WAV biçimi (tag=%d bits=%d)" % (tag, bits))
    return pcm, ch


def load_track(path):
    """Ses dosyasını (n,2) float32 diziye okur.

    Her formatı macOS'un afconvert'i ile geçici kanonik WAV'a çeviriyoruz
    (16-bit, stereo, SR). Böylece mp3/m4a/flac için ek paket gerekmiyor
    ve örnekleme hızı dönüşümünü de o yapıyor.
    """
    if shutil.which("afconvert") is None:
        raise RuntimeError("afconvert bulunamadı (macOS dışı sistem?).")
    tmp = os.path.join(tempfile.mkdtemp(), "conv.wav")
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@%d" % SR, "-c", "2", path, tmp],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("afconvert başarısız: %s" % r.stderr.strip()[:300])
    raw, ch = read_wav(tmp)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    a = a.reshape(-1, ch)
    if ch == 1:
        a = np.column_stack([a[:, 0], a[:, 0]])
    os.remove(tmp)
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak * 0.5                      # efektler için tepe payı bırak
    return np.ascontiguousarray(a)


def find_track(argv):
    if len(argv) > 1:
        return argv[1]
    for pat in ("*.mp3", "*.wav", "*.m4a", "*.aiff", "*.flac", "*.aac"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


# ---------------- DSP ----------------
def low_shelf(db, f0=150.0):
    """RBJ cookbook low-shelf biquad. Bas kontrolü için."""
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * f0 / SR
    c, sn = np.cos(w0), np.sin(w0)
    alpha = sn / 2 * np.sqrt(2.0)
    sq = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) - (A - 1) * c + sq),
                  2 * A * ((A - 1) - (A + 1) * c),
                  A * ((A + 1) - (A - 1) * c - sq)])
    a = np.array([(A + 1) + (A - 1) * c + sq,
                  -2 * ((A - 1) + (A + 1) * c),
                  (A + 1) + (A - 1) * c - sq])
    return b / a[0], a / a[0]


class Engine:
    """sounddevice geri çağrısında çalışan ses motoru.

    Parametreler ana iş parçacığından `target` sözlüğüne yazılır; geri çağrı
    her blokta `cur` değerlerini hedefe doğru yumuşatarak yaklaştırır.
    """

    def __init__(self, track):
        self.track = track
        self.n = len(track)
        self.pos = 0.0
        self.paused = False
        self.target = {k: nv for k, _, nv in ALL_FX}
        self.cur = dict(self.target)
        self.zi_lp = np.zeros((2, 2))
        self.zi_hp = np.zeros((2, 2))
        self.zi_bs = np.zeros((2, 2))
        self.dly_n = int(0.25 * SR)
        self.dly = np.zeros((self.dly_n, 2), dtype=np.float32)
        self.dly_i = 0
        self.trem_ph = 0.0
        self.crush_env = 0.0        # bitcrush seviye takipçisi
        self.flt_gain = 1.0         # filtre seviye telafisi
        self.underruns = 0

    # --- parametre eşlemeleri (0..1 -> fiziksel birim) ---
    @staticmethod
    def map_speed(v):
        return float(2.0 ** ((v - 0.5) * 2.0 * FX_SCALE))   # FX_SCALE=1.5 -> 0.35x .. 2.83x

    def callback(self, out, frames, time_info, status):
        if status:
            self.underruns += 1
        if self.paused:
            out.fill(0)
            return

        prev_speed = self.cur["speed"]
        for k in self.cur:                           # yumuşatma
            self.cur[k] += (self.target[k] - self.cur[k]) * SMOOTH
        c = self.cur

        # --- varispeed okuma: blok içinde hızı doğrusal rampala ---
        s0, s1 = self.map_speed(prev_speed), self.map_speed(c["speed"])
        inc = np.linspace(s0, s1, frames, dtype=np.float64)
        idx = (self.pos + np.cumsum(inc) - inc) % self.n
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)[:, None]
        i1 = (i0 + 1) % self.n
        x = self.track[i0] * (1 - frac) + self.track[i1] * frac
        self.pos = (self.pos + float(inc.sum())) % self.n

        # --- bas (low shelf) ---
        if BASS_BIPOLAR:
            bass_db = (c["bass"] - 0.5) * 2.0 * BASS_DB_MAX   # -18 .. +18 dB
        else:
            bass_db = c["bass"] * BASS_DB_MAX
        if abs(bass_db) > 0.05:
            b, a = low_shelf(bass_db)
            x, self.zi_bs = lfilter(b, a, x, axis=0, zi=self.zi_bs)

        # --- lowpass / highpass, seviye telafisiyle ---
        filtering = c["lowpass"] > 0.01 or c["highpass"] > 0.01
        pre_rms = float(np.sqrt(np.mean(x * x))) if filtering else 0.0

        if c["lowpass"] > 0.01:
            fc = _LP_HI * (LP_LO / _LP_HI) ** c["lowpass"]
            b, a = butter(2, min(fc / (SR / 2), 0.99), btype="low")
            x, self.zi_lp = lfilter(b, a, x, axis=0, zi=self.zi_lp)

        if c["highpass"] > 0.01:
            fc = _HP_LO * (HP_HI / _HP_LO) ** c["highpass"]
            b, a = butter(2, max(fc / (SR / 2), 1e-4), btype="high")
            x, self.zi_hp = lfilter(b, a, x, axis=0, zi=self.zi_hp)

        if filtering:
            post_rms = float(np.sqrt(np.mean(x * x)))
            want = min(max(pre_rms / max(post_rms, 1e-7), 1.0), FLT_MAKEUP_MAX)
        else:
            want = 1.0
        self.flt_gain += (want - self.flt_gain) * FLT_MAKEUP_SMOOTH
        x = x * self.flt_gain

        # --- bitcrush ---
        if c["crush"] > 0.01:
            bits = 16.0 - c["crush"] * (16.0 - CRUSH_BITS_MIN)
            lv = 2 ** (bits - 1)
            pk = float(np.max(np.abs(x)))
            self.crush_env = max(pk, self.crush_env * CRUSH_RELEASE)
            pre = CRUSH_TARGET / max(self.crush_env, CRUSH_FLOOR)
            x = np.round(x * pre * lv) / (lv * pre)

        # --- tremolo ---
        if c["trem"] > 0.01:
            ph = self.trem_ph + 2 * np.pi * 8.0 * np.arange(frames) / SR
            depth = c["trem"] * TREM_DEPTH_MAX
            mod = 1 - depth * 0.5 * (1 - np.sin(ph))
            # Modülatörün ortalaması (1 - depth/2); bölmezsek tremolo saf ses
            # kısılması gibi duyuluyor (ölçtük: -4.2 dB, ton değişimi sıfır).
            mod = mod / (1.0 - depth * 0.5)
            x = x * mod[:, None].astype(np.float32)
            self.trem_ph = float((ph[-1] + 2 * np.pi * 8.0 / SR) % (2 * np.pi))

        # --- delay (dairesel tampon, sabit 250 ms) ---
        mix = c["delay"] * DELAY_MIX_MAX
        rd = (self.dly_i + np.arange(frames)) % self.dly_n
        echo = self.dly[rd].copy()
        self.dly[rd] = (x + echo * DELAY_FB).astype(np.float32)
        self.dly_i = int((self.dly_i + frames) % self.dly_n)
        if mix > 0.001:
            x = x * (1 - mix * 0.5) + echo * mix

        # --- ses, sonra yumuşak doyma ---
        # Bastan gelen küresel kısma kaldırıldı: ortayı/tizi düşürüp bas
        # artışını örtüyordu. Aşırı yüklenmeyi artık limitleyici karşılıyor;
        # sert kırpmanın çıtırtısı yerine yumuşak doyma veriyor.
        y = x * (c["vol"] * 0.9)
        a = np.abs(y)
        if a.max() > LIMIT_KNEE:
            soft = LIMIT_KNEE + (1.0 - LIMIT_KNEE) * np.tanh((a - LIMIT_KNEE) / (1.0 - LIMIT_KNEE))
            y = np.where(a > LIMIT_KNEE, np.sign(y) * soft, y)
        out[:] = np.clip(y, -1.0, 1.0).astype(np.float32)


# ---------------- El takibi ----------------
def px_dist(a, b, w, h):
    return (((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2) ** 0.5


def pinch_finger(lm, w, h):
    ref = px_dist(lm[0], lm[9], w, h)
    if ref < 1e-6:
        return None
    best, best_ratio = None, PINCH_RATIO
    for i, tip in enumerate(FINGER_TIPS):
        r = px_dist(lm[THUMB_TIP], lm[tip], w, h) / ref
        if r < best_ratio:
            best, best_ratio = i, r
    return best


def hand_value(lm):
    """El yüksekliğini 0..1'e çevirir (yukarı = 1). Avuç merkezi kullanılıyor."""
    y = lm[9].y
    return float(np.clip((Y_BOTTOM - y) / (Y_BOTTOM - Y_TOP), 0.0, 1.0))


def user_hand(name):
    """Görüntüyü aynaladığımız için MediaPipe'ın etiketi ters gelir."""
    return "Right" if name == "Left" else "Left"


def apply_hand(eng, grab, hid, finger, val, fx):
    """Bir elin durumunu efekt hedeflerine yazar.

    Seçili olmayan efektlere hiç dokunulmaz, böylece son değerlerinde kalırlar.
    Göreli devralmada yakalama anındaki (yükseklik, değer) çifti saklanır ve
    efekt o noktadan elin hareketi kadar kaydırılır; bu yüzden yakalarken
    değer zıplamaz.
    """
    if finger is None:
        grab[hid] = None                    # dokunma bitti, efekt olduğu yerde kalıyor
        return
    key = fx[finger][0]
    if grab[hid] is None or grab[hid][0] != finger:
        grab[hid] = (finger, val, eng.target[key])       # yakalama noktası
    if RELATIVE_TAKEOVER:
        _, y0, v0 = grab[hid]
        v = v0 + (val - y0)
    else:
        v = val
    # Her iki kolda da kırp: aralık dışı bir değer doğrudan DSP'ye giderse
    # ses seviyesi veya hız saçmalayabilir.
    eng.target[key] = float(np.clip(v, 0.0, 1.0))


def draw_hand(frame, lm, color):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_panel(frame, x0, y0, title, fx, active, cur, color):
    cv2.putText(frame, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for i, (key, label, _) in enumerate(fx):
        y = y0 + i * 34
        on = (i == active)
        cv2.rectangle(frame, (x0, y), (x0 + 250, y + 28),
                      (60, 60, 60) if on else (34, 34, 34), -1)
        v = cur[key]
        cv2.rectangle(frame, (x0, y), (x0 + int(250 * v), y + 28),
                      color if on else (80, 80, 80), -1)
        cv2.rectangle(frame, (x0, y), (x0 + 250, y + 28), (110, 110, 110), 1)
        cv2.putText(frame, "%-9s %3d%%" % (label, int(v * 100)), (x0 + 8, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255) if on else (190, 190, 190), 1)


def main():
    path = find_track(sys.argv)
    if path:
        print("Parça yükleniyor:", path)
        track = load_track(path)
        name = os.path.basename(path)
    else:
        print("Ses dosyası bulunamadı -> sentetik test loop'u kullanılıyor.")
        print("Kendi parçan için: venv/bin/python handdj.py parca.mp3")
        track = synth_loop()
        name = "sentetik loop"
    print("süre: %.1f sn" % (len(track) / SR))

    eng = Engine(track)
    stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=2,
                             dtype="float32", callback=eng.callback)
    stream.start()

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
        print("Kamera açılamadı. Sistem Ayarları > Gizlilik > Kamera izni gerekiyor.")
        stream.stop()
        return

    start, last_ts = time.time(), -1
    grab = [None, None]          # el başına (parmak, yakalama yüksekliği, yakalama değeri)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - start) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        left, right = (None, 0.0), (None, 0.0)
        for lm, handed in zip(result.hand_landmarks, result.handedness):
            if user_hand(handed[0].category_name) == "Left":
                left = (pinch_finger(lm, w, h), hand_value(lm))
                draw_hand(frame, lm, (255, 150, 0))
            else:
                right = (pinch_finger(lm, w, h), hand_value(lm))
                draw_hand(frame, lm, (0, 120, 255))

        # Seçili efekt elin hareketini alır; seçili OLMAYANLAR son değerinde kalır.
        for hid, (finger, val), fx in ((0, left, LEFT_FX), (1, right, RIGHT_FX)):
            apply_hand(eng, grab, hid, finger, val, fx)

        draw_panel(frame, 20, 90, "SOL EL", LEFT_FX, left[0], eng.cur, (255, 150, 0))
        draw_panel(frame, 300, 90, "SAG EL", RIGHT_FX, right[0], eng.cur, (0, 120, 255))

        spd = Engine.map_speed(eng.cur["speed"])
        hud = "%s | hiz %.2fx | r=sifirla%s" % (
            name[:24], spd, "  [DURAKLADI]" if eng.paused else "")
        cv2.putText(frame, hud, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if eng.underruns:
            cv2.putText(frame, "ses atlamasi: %d" % eng.underruns, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        cv2.imshow("El DJ", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            eng.paused = not eng.paused
        if k == ord("r"):                   # kalıcı değerler için geri dönüş yolu
            for key, _, nv in ALL_FX:
                eng.target[key] = nv
            grab[0] = grab[1] = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    stream.stop()
    stream.close()


if __name__ == "__main__":
    main()
