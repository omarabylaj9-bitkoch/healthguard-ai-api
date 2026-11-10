"""HealthGuard AI — признаки и алгоритмы (копия из ноутбука HealthGuard_AI_MVP.ipynb)."""
import numpy as np, pandas as pd
from scipy import signal as sps, stats


# ---------------- общее ----------------
def robust_baseline(x):
    """Личная норма: медиана и робастное SD (1.4826·MAD)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    med = np.median(x); sd = 1.4826 * np.median(np.abs(x - med))
    return med, (sd if sd > 1e-9 else (np.std(x) + 1e-9))

# ---------------- Heart ----------------
HEART_CALIB_BEATS = 300
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "aAJS"}, **{s: "V" for s in "VE"}, "F": "F"}
HEART_FEATS = ["RR_до/норма", "RR_после/норма", "RR_ср10/норма", "RR_до/RR_ср10", "RR_после/RR_до",
               "(RR_до−RR_ср10)/норма", "RR_до/RR_ср5мин", "(RR_после−RR_до)/норма", "RR_до2/норма", "RR_после2/норма",
               "размах QRS", "ширина QRS", "сходство с шаблоном (весь удар)", "сходство с шаблоном (QRS)",
               "«длина» QRS", "макс QRS", "мин QRS", "сходство P-волны с шаблоном", "размах P-волны",
               "макс P-волны", "отличие QRS от шаблона"]

def beat_features(sig, samples, symbols, fs):
    """Признаки удара: ритм (RR-интервалы вокруг удара) + форма (QRS и P-волна) в сравнении
    с личным шаблоном человека. Возвращает «сырые» признаки — личная нормировка ниже."""
    b, a = sps.butter(2, [0.5/(fs/2), 40/(fs/2)], "band"); x = sps.filtfilt(b, a, sig)
    rr = np.diff(samples) / fs; mean_rr = np.median(rr[:HEART_CALIB_BEATS])
    pre, post = int(0.30*fs), int(0.40*fs)
    q0, q1 = int(.06*fs), int(.10*fs)                      # окно QRS
    p0, p1 = int(.25*fs), int(.08*fs)                      # окно P-волны (до QRS)
    idx = [i for i in range(10, len(samples) - 2)
           if AAMI.get(symbols[i]) and samples[i] - pre >= 0 and samples[i] + post < len(x)]
    if not idx: return np.zeros((0, len(HEART_FEATS))), np.array([])
    W = np.stack([x[samples[i]-pre: samples[i]+post] for i in idx])
    W -= np.median(W[:, pre-int(.10*fs): pre-int(.05*fs)], 1, keepdims=True)   # изолиния перед QRS
    tmpl = np.median(W[:HEART_CALIB_BEATS], 0)              # личный шаблон из калибровки
    amp = np.median(np.ptp(W[:HEART_CALIB_BEATS, pre-q0:pre+q1], 1)) + 1e-6
    tq, tp = tmpl[pre-q0:pre+q1], tmpl[pre-p0:pre-p1]
    feats, labels = [], []
    for k, i in enumerate(idx):
        pre_rr, post_rr = rr[i-1], rr[i]; loc = np.mean(rr[i-10:i]); loc5 = np.mean(rr[max(0, i-300):i])
        w = W[k]; qrs = w[pre-q0: pre+q1]; pw = w[pre-p0: pre-p1]
        width = np.sum(np.abs(qrs) > 0.5 * np.max(np.abs(qrs))) / fs
        feats.append([pre_rr/mean_rr, post_rr/mean_rr, loc/mean_rr, pre_rr/loc, post_rr/pre_rr,
                      (pre_rr - loc)/mean_rr, pre_rr/loc5, (post_rr - pre_rr)/mean_rr, rr[i-2]/mean_rr, rr[i+1]/mean_rr,
                      np.ptp(qrs)/amp, width, np.corrcoef(w, tmpl)[0, 1], np.corrcoef(qrs, tq)[0, 1],
                      np.sum(np.abs(np.diff(qrs)))/amp, np.max(qrs)/amp, np.min(qrs)/amp,
                      np.corrcoef(pw, tp)[0, 1], np.ptp(pw)/amp, np.max(pw)/amp,
                      np.sum((qrs - tq)**2) / (np.sum(tq**2) + 1e-9)])
        labels.append(AAMI[symbols[i]])
    return np.nan_to_num(np.array(feats)), np.array(labels)

def personal_norm(F, n0=HEART_CALIB_BEATS):
    """Личная нормировка: (признак − медиана калибровки) / робастный разброс калибровки."""
    med = np.median(F[:n0], 0); mad = 1.4826 * np.median(np.abs(F[:n0] - med), 0) + 1e-6
    return (F - med) / mad

BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ")

def _mwi(sig, fs):
    x = sps.filtfilt(*sps.butter(2, [5/(fs/2), 15/(fs/2)], "band"), sig)
    w = int(0.15*fs); return np.convolve(np.gradient(x) ** 2, np.ones(w)/w, "same")

def detect_beats(sig, fs, frac=0.3, win=5.0):
    mwi = _mwi(sig, fs); pk, _ = sps.find_peaks(mwi, distance=int(0.2*fs))
    if len(pk) < 3: return np.array([], int)
    h = mwi[pk]; W = int(win*fs); lo = np.searchsorted(pk, pk - W); hi = np.searchsorted(pk, pk + W)
    ref = np.array([np.percentile(h[l:u], 80) for l, u in zip(lo, hi)]); nz = np.array([np.percentile(h[l:u], 20) for l, u in zip(lo, hi)])
    thr = nz + frac * (ref - nz); out = pk[h > thr]
    if len(out) > 3:                                                    # searchback: длинная пауза → ищем пропущенный удар
        rr = np.diff(out); add = []
        for i, d in enumerate(rr):
            if d > 1.66 * np.median(rr[max(0, i-8): i+1]):
                cand = pk[(pk > out[i] + 0.2*fs) & (pk < out[i+1] - 0.2*fs)]
                if len(cand):
                    c = cand[np.argmax(mwi[cand])]
                    if mwi[c] > 0.5 * thr[np.searchsorted(pk, c)]: add.append(c)
        out = np.sort(np.r_[out, add]).astype(int)
    r = int(0.1*fs); bp = sps.filtfilt(*sps.butter(2, [0.5/(fs/2), 40/(fs/2)], "band"), sig)
    return np.array([max(0, p - r) + np.argmax(np.abs(bp[max(0, p - r): p + r])) for p in out])

def match_beats(det, ref, fs, tol=0.15):
    t = int(tol*fs); used = np.zeros(len(det), bool); tp = 0; j = 0
    for r in ref:
        while j < len(det) and det[j] < r - t: j += 1
        k = j
        while k < len(det) and det[k] <= r + t:
            if not used[k]: used[k] = True; tp += 1; break
            k += 1
    return np.array([tp, len(ref) - tp, len(det) - tp])

def ecg_quality(sig, fs, det, win_s=10, corr_thr=0.8):
    """Окно «хорошее», если сокращения похожи на свой шаблон (медианная корреляция ≥ 0.8) и ритм 18–240 уд/мин."""
    bp = sps.filtfilt(*sps.butter(2, [0.5/(fs/2), 40/(fs/2)], "band"), sig); w = int(0.1*fs); out = []
    for i in range(int(len(sig) / (win_s*fs))):
        a, b = int(i*win_s*fs), int((i+1)*win_s*fs); d = det[(det >= a + w) & (det < b - w)]
        c = np.median([np.corrcoef(x, np.median(np.stack([bp[r-w:r+w] for r in d]), 0))[0, 1]
                       for x in np.stack([bp[r-w:r+w] for r in d])]) if len(d) >= 3 else 0.0
        out.append((a, b, bool(3 <= len(d) <= 40 and c >= corr_thr)))
    return out

def sqi_feats(x0, x1, fs, d0, d1, a, b):
    s0 = x0[a:b]; f, p = sps.welch(s0 - s0.mean(), fs=fs, nperseg=512); e = lambda lo, hi: p[(f >= lo) & (f < hi)].sum()
    q0 = d0[(d0 >= a) & (d0 < b)]; q1 = d1[(d1 >= a) & (d1 < b)]; m = match_beats(q0, q1, fs)
    rr = np.diff(q0) / fs
    return [2*m[0] / max(2*m[0] + m[1] + m[2], 1), e(5, 15) / (e(5, 40) + 1e-12), e(1, 40) / (e(0, 40) + 1e-12), stats.kurtosis(s0),
            len(q0), rr.std() / rr.mean() if len(rr) > 2 else 1.0, e(40, fs/2) / (p.sum() + 1e-12), abs(len(q0) - len(q1))]

# ---------------- Sleep ----------------
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 25), (25, 35)]

def chan_feats(x, fs):
    f, p = sps.welch(x, fs=fs, nperseg=int(4*fs), axis=-1)
    tot = p[:, (f >= 0.5) & (f <= 35)].sum(-1) + 1e-20
    bp = np.stack([p[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], 1) / tot[:, None]
    cum = np.cumsum(p[:, (f >= 0.5) & (f <= 35)], 1) / tot[:, None]; fr = f[(f >= 0.5) & (f <= 35)]
    sef50 = fr[np.argmax(cum >= 0.5, 1)]; sef95 = fr[np.argmax(cum >= 0.95, 1)]
    d1 = np.diff(x, axis=-1); d2 = np.diff(d1, axis=-1)
    act = x.var(-1) + 1e-20; mob = np.sqrt(d1.var(-1) / act); comp = np.sqrt(d2.var(-1) / (d1.var(-1) + 1e-20)) / (mob + 1e-9)
    zc = (np.diff(np.sign(x), axis=-1) != 0).mean(-1)
    delta, theta, alpha, sigma, beta = bp[:, :2].sum(1), bp[:, 2], bp[:, 3], bp[:, 4], bp[:, 5]
    pe = -(bp * np.log(bp + 1e-12)).sum(1)
    return np.column_stack([np.log(bp + 1e-9), np.log(tot), sef50, sef95, np.log(act), mob, comp, zc,
                            stats.kurtosis(x, axis=-1), stats.skew(x, axis=-1), np.log(np.ptp(x, -1) + 1e-12),
                            np.log((delta + 1e-9) / (beta + 1e-9)), np.log((theta + 1e-9) / (alpha + 1e-9)),
                            np.log((sigma + 1e-9) / (delta + 1e-9)), pe])

def sleep_context(F, k=3):
    parts = [F]
    for d in range(1, k + 1): parts += [np.roll(F, d, 0), np.roll(F, -d, 0)]
    k5 = pd.DataFrame(F).rolling(5, center=True, min_periods=1).mean().values
    k15 = pd.DataFrame(F).rolling(15, center=True, min_periods=1).mean().values
    return np.hstack(parts + [k5, k15, F - k15])

def sleep_personal(F):
    """Личная норма ночи (без меток): признаки относительно медианы/MAD этой же ночи этого человека."""
    med = np.median(F, 0); return (F - med) / (1.4826 * np.median(np.abs(F - med), 0) + 1e-6)

def sleep_long_ctx(F):
    """Длинный контекст 15 / 30 / 60 мин — положение внутри цикла сна (~90 мин)."""
    out = [pd.DataFrame(F).rolling(w, center=True, min_periods=1).mean().values for w in (31, 61, 121)]
    return np.hstack(out + [pd.DataFrame(F).rolling(31, center=True, min_periods=1).std().fillna(0).values])

HEADBAND_COLS = list(range(21)) + [63, 64]

# ---------------- Activity ----------------
def har_context(X, g, k=2):
    """Соседние окна того же человека (окна идут подряд во времени): среднее ±k окон и разность с ним.
    Сидит/стоит по одному окну 2.56 с почти не отличить — по 12 секундам контекста отличаются."""
    out = np.zeros_like(X)
    for s in np.unique(g):
        i = np.where(g == s)[0]; out[i] = pd.DataFrame(X[i]).rolling(2*k + 1, center=True, min_periods=1).mean().values
    return np.hstack([X, out, X - out])

def smooth_proba(P, g, k=2):
    """Сглаживание вероятностей по времени: активность не меняется каждые 2.5 секунды."""
    out = np.zeros_like(P)
    for s in np.unique(g):
        i = np.where(g == s)[0]; out[i] = pd.DataFrame(P[i]).rolling(2*k + 1, center=True, min_periods=1).mean().values
    return out

def mag_feats(m, fs):
    f, p = sps.welch(m - m.mean(), fs=fs, nperseg=min(len(m), 128)); tot = p.sum() + 1e-12
    ac = np.correlate(m - m.mean(), m - m.mean(), "full")[len(m)-1:]; ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.3*fs), min(int(1.5*fs), len(ac) - 1); k = lo + np.argmax(ac[lo:hi]) if hi > lo else 0
    bands = [p[(f >= a) & (f < b)].sum() / tot for a, b in ((0, 1), (1, 3), (3, 6), (6, 12))]
    return [m.mean(), m.std(), *np.percentile(m, [10, 50, 90]), np.ptp(m), np.mean(m**2), np.abs(np.diff(m)).mean(),
            f[p.argmax()], *bands, ac[k], k / fs]

def highpass(x, fs, fc=0.3):
    b, a = sps.butter(2, fc/(fs/2), "high"); return sps.filtfilt(b, a, x, axis=0)

# ---------------- Mental ----------------
FS_W = 700
def _bp(x, lo, hi, fs):
    b, a = sps.butter(2, [lo/(fs/2), hi/(fs/2)], "band"); return sps.filtfilt(b, a, x)

def _lp(x, hi, fs):
    b, a = sps.butter(2, hi/(fs/2)); return sps.filtfilt(b, a, x)

def _hrv(ibi):
    ibi = ibi[(ibi > 300) & (ibi < 2000)]
    if len(ibi) < 15: return [np.nan] * 4
    d = np.diff(ibi); return [60000/ibi.mean(), ibi.std(), np.sqrt(np.mean(d**2)), np.mean(np.abs(d) > 50)]

def _eda(e, fs):
    sc, _ = sps.find_peaks(e, prominence=0.01 if fs <= 8 else 0.02)
    return [e.mean(), e.std(), np.polyfit(np.arange(len(e)), e, 1)[0] * fs * 60, len(sc), np.ptp(e)]

def wesad_windows(d, win=60, step=15):
    """Признаки по окнам: грудь (14) и браслет (13). Метки: 1 спокойствие, 2 стресс (TSST), 3 радость."""
    lab = np.asarray(d["label"]).ravel(); ch = d["signal"]["chest"]; wr = d["signal"]["wrist"]
    ecg = _bp(ch["ECG"].ravel(), 5, 20, FS_W); eda_c = _lp(ch["EDA"].ravel(), 1, FS_W); resp = ch["Resp"].ravel()
    temp_c = ch["Temp"].ravel(); acc_c = np.linalg.norm(ch["ACC"], axis=1)
    bvp = _bp(np.asarray(wr["BVP"]).ravel(), 0.7, 3.5, 64); eda_w = np.asarray(wr["EDA"]).ravel(); temp_w = np.asarray(wr["TEMP"]).ravel()
    acc_w = np.linalg.norm(np.asarray(wr["ACC"], float), axis=1)
    rows_c, rows_w, ys = [], [], []; W, S = win * FS_W, step * FS_W
    for st in range(0, len(lab) - W, S):
        seg = lab[st:st+W]; v, c = np.unique(seg, return_counts=True); maj = v[c.argmax()]
        if maj not in (1, 2, 3) or c.max() < 0.9 * W: continue
        t0, t1 = st / FS_W, (st + W) / FS_W
        e = ecg[st:st+W]; pk, _ = sps.find_peaks(e, distance=int(.33*FS_W), height=np.percentile(e, 95) * .5)
        f, px = sps.welch(resp[st:st+W:70] - resp[st:st+W:70].mean(), fs=10, nperseg=512); bnd = (f > .1) & (f < .7)
        rows_c.append(_hrv(np.diff(pk) / FS_W * 1000) + _eda(eda_c[st:st+W:175], 4) +
                      [f[bnd][px[bnd].argmax()] * 60, resp[st:st+W].std(), temp_c[st:st+W].mean(),
                       np.polyfit(np.arange(0, W, 700), temp_c[st:st+W:700], 1)[0], acc_c[st:st+W].std()])
        b = bvp[int(t0*64):int(t1*64)]; pk, _ = sps.find_peaks(b, distance=int(.33*64), prominence=np.std(b) * .3)
        ew = eda_w[int(t0*4):int(t1*4)]; tw = temp_w[int(t0*4):int(t1*4)]; aw = acc_w[int(t0*32):int(t1*32)]
        rows_w.append(_hrv(np.diff(pk) / 64 * 1000) + (_eda(ew, 4) if len(ew) > 8 else [np.nan]*5) +
                      [tw.mean(), np.polyfit(np.arange(len(tw)), tw, 1)[0] if len(tw) > 4 else np.nan, aw.std(), np.ptp(aw)])
        ys.append(int(maj))
    return np.array(rows_c, float), np.array(rows_w, float), np.array(ys)

def personal_start(X, n0=40):
    """Личная норма = первые 40 окон (≈10 мин спокойного начала записи). Остальная запись в норму не входит."""
    ref = X[:n0]; return np.nan_to_num((X - np.nanmedian(ref, 0)) / (np.nanstd(ref, 0) + 1e-9))

MENTAL_FEATS = (["chest_" + n for n in ["HR", "SDNN", "RMSSD", "pNN50", "EDA_mean", "EDA_std", "EDA_slope", "SCR_count", "EDA_range",
                                       "Resp_rate", "Resp_std", "Temp_mean", "Temp_slope", "ACC_std"]] +
                ["wrist_" + n for n in ["HR", "SDNN", "RMSSD", "pNN50", "EDA_mean", "EDA_std", "EDA_slope", "SCR_count", "EDA_range",
                                       "Temp_mean", "Temp_slope", "ACC_std", "ACC_range"]])

# ---------------- Unified Engine ----------------
SIGNALS = {  # сигнал: (модуль, вредное направление, мин. SD, единица, название)
    "resting_hr":  ("Heart", +1, 2.0, "уд/мин", "пульс в покое"),
    "rmssd":       ("Heart", -1, 4.0, "мс", "вариабельность ритма (RMSSD)"),
    "ectopic_pct": ("Heart", +1, 0.3, "%", "доля внеочередных сокращений"),
    "sleep_h":     ("Sleep", -1, 0.4, "ч", "длительность сна"),
    "sleep_eff":   ("Sleep", -1, 2.5, "%", "эффективность сна"),
    "active_min":  ("Activity", -1, 8.0, "мин", "активное время / тренировки"),
    "rad_mSv":     ("Radiation", +1, 0.03, "м(Зв|Гр)/сут", "суточная доза"),
    "stress_frac": ("Mental", +1, 0.05, "", "доля времени с признаками стресса"),
    "pvt_rt":      ("Mental", +1, 12.0, "мс", "время реакции (PVT)"),
}

LEVELS = ["Normal", "Attention", "Warning", "Critical"]

ADVICE = {
    "Heart": "Отдохните 15 минут и повторите измерение; если изменение сохранится — данные будут переданы бортовому врачу.",
    "Sleep": "Постарайтесь сегодня лечь вовремя; команда может пересмотреть расписание сна.",
    "Activity": "Вы выполняете меньше тренировок, чем обычно — проверьте план упражнений на сегодня.",
    "Radiation": "Следуйте процедуре миссии при повышенной радиации; центр управления уведомлён.",
    "Mental": "Сделайте перерыв; при желании поговорите с психологической поддержкой миссии.",
}

class UnifiedHealthEngine:
    def __init__(self, baseline_days=14, smooth=0.5, hard_rules=None):
        self.bd, self.alpha = baseline_days, smooth
        self.hard = hard_rules or {"ectopic_pct": 5.0}
        self.event_signals = ("rad_mSv", "ectopic_pct")
    def fit_baseline(self, df):
        self.base = {}
        for s, (_, _, minsd, *_r) in SIGNALS.items():
            if s in df:
                m, sd = robust_baseline(df[s].iloc[:self.bd]); self.base[s] = (m, max(sd, minsd))
        return self
    def assess(self, df, rad_events_days=()):
        z = pd.DataFrame(index=df.index)
        for s, (mod, sign, *_r) in SIGNALS.items():
            if s not in self.base: continue
            m, sd = self.base[s]
            a_s = 1.0 if s in self.event_signals else self.alpha   # событийные сигналы — без сглаживания (нет «хвоста» тревоги)
            z[s] = (sign * (df[s] - m) / sd).ewm(alpha=a_s, ignore_na=True).mean()
        thr = pd.Series({c: (3.0 if c in self.event_signals else 2.0) for c in z.columns})   # одиночные всплески — порог выше
        att = z.ge(thr, axis=1); sig_lvl = att.astype(int) + z.ge(thr + 1, axis=1).astype(int)
        for s in sig_lvl:                                    # ≥3 дня подряд Attention → Warning
            run = att[s].astype(int); run = run.groupby((run == 0).cumsum()).cumsum()
            sig_lvl.loc[run >= 3, s] = sig_lvl.loc[run >= 3, s].clip(lower=2)
        mod_lvl = pd.DataFrame({mod: sig_lvl[[s for s in sig_lvl if SIGNALS[s][0] == mod]].max(1)
                                for mod in sorted({v[0] for v in SIGNALS.values()})})
        overall = mod_lvl.max(1).copy()
        overall[(mod_lvl >= 1).sum(1) >= 3] = overall.clip(lower=2)
        overall[(mod_lvl >= 2).sum(1) >= 3] = 3
        hard = pd.Series(False, index=df.index)
        for s, thr in self.hard.items():
            if s in df: hard |= df[s] >= thr
        self._rad_days = set(rad_events_days)
        hard |= df.index.isin(list(rad_events_days)); overall[hard] = 3
        self.z, self.sig_lvl, self.mod_lvl, self.overall, self.hard_flags = z, sig_lvl, mod_lvl, overall, hard
        return overall
    def explain(self, df, day):
        lvl = int(self.overall.loc[day]); msgs = []
        quality = 1 - df.loc[day, list(self.base)].isna().mean()
        for s in self.base:
            if self.sig_lvl.loc[day, s] >= 1 or (s in self.hard and df.loc[day, s] >= self.hard[s]):
                mod, _, _, unit, title = SIGNALS[s]; m, _ = self.base[s]; v = df.loc[day, s]
                pct = 100*(v - m)/abs(m) if m else np.nan
                msgs.append(dict(module=mod, signal=title, value=round(float(v), 2), baseline=round(float(m), 2),
                                 unit=unit, change_pct=round(float(pct), 1), z=round(float(self.z.loc[day, s]), 1)))
        if day in getattr(self, "_rad_days", ()): msgs.append(dict(module="Radiation", signal="радиационное событие"))
        return dict(day=int(day), level=LEVELS[lvl], data_quality=round(float(quality), 2), findings=msgs,
                    reason=("жёсткое правило миссии" if self.hard_flags.loc[day] else
                            f"{(self.mod_lvl.loc[day] >= 1).sum()} модул(я/ей) вне личной нормы"))
    def notify(self, df, day):
        e = self.explain(df, day)
        if e["level"] == "Normal": return None
        lines = [f"[{e['level']}] День {e['day']}. Это не диагноз — система заметила изменения относительно вашей нормы:"]
        for f in e["findings"]:
            if "value" in f:
                lines.append(f"  • {f['signal']}: {f['value']} {f['unit']} (ваша норма {f['baseline']}, {f['change_pct']:+.0f}%) — {f['module']} AI")
        for mod in sorted({f["module"] for f in e["findings"]}): lines.append("  → " + ADVICE[mod])
        if e["level"] in ("Warning", "Critical"): lines.append("  → Медицинская команда получила уведомление для проверки.")
        lines.append(f"  (качество данных: {e['data_quality']:.0%}; причина уровня: {e['reason']})")
        return "\n".join(lines)


# ================= Обёртки для API: из сырых сигналов в признаки =================
def to_360(sig, fs):
    """Модели Heart обучены на 360 Гц (MIT-BIH) — приводим частоту."""
    sig = np.asarray(sig, float)
    if fs == 360: return sig, 360
    from fractions import Fraction
    fr = Fraction(360, int(round(fs))).limit_denominator(1000)
    return sps.resample_poly(sig, fr.numerator, fr.denominator), 360

def headband_epoch_features(eeg_fpz, eog, fs):
    """Повязка: лобный канал ЭЭГ (Fpz-Cz, вольты) + движения глаз (вольты), эпохи по 30 с → 23 признака на эпоху."""
    n = int(len(eeg_fpz) // (30 * fs)); L = int(30 * fs)
    X = np.asarray(eeg_fpz[:n * L], float).reshape(n, L); E = np.asarray(eog[:n * L], float).reshape(n, L)
    F = chan_feats(X, fs)
    f, p = sps.welch(E, fs=fs, nperseg=int(4 * fs), axis=-1)
    Fe = np.column_stack([np.log(p[:, (f >= 0.3) & (f < 2)].sum(-1) + 1e-20), np.log(p[:, (f >= 2) & (f < 5)].sum(-1) + 1e-20)])
    return np.nan_to_num(np.hstack([F, Fe]))

def headband_model_input(F23):
    t = np.arange(len(F23)) / max(len(F23), 1); Fp = sleep_personal(F23)
    return np.hstack([sleep_context(F23), sleep_context(Fp), sleep_long_ctx(Fp), t[:, None]]).astype(np.float32)

def wrist_epoch_features(t_acc, acc_xyz, t_hr, hr):
    """Браслет (как Apple Watch в Walch et al. 2019): ускорение в g, время в секундах от начала ночи; пульс уд/мин."""
    t_acc = np.asarray(t_acc, float); o = np.argsort(t_acc, kind="stable"); t_acc = t_acc[o]
    mag = np.linalg.norm(np.asarray(acc_xyz, float)[o], axis=1)
    t_hr = np.asarray(t_hr, float); o2 = np.argsort(t_hr, kind="stable"); t_hr, hr = t_hr[o2], np.asarray(hr, float)[o2]
    rows, starts = [], []
    for t in np.arange(max(0.0, t_acc[0]), t_acc[-1] - 30, 30):
        a0, a1 = np.searchsorted(t_acc, [t, t + 30]); m = mag[a0:a1]
        h0, h1 = np.searchsorted(t_hr, [t - 30, t + 60]); h = hr[h0:h1]
        if len(m) < 50: continue
        dm = np.abs(np.diff(m))
        rows.append([m.std(), dm.mean(), np.percentile(dm, 95), (dm > 0.01).mean(), np.ptp(m),
                     h.mean() if len(h) else np.nan, h.std() if len(h) > 1 else np.nan,
                     np.polyfit(np.arange(len(h)), h, 1)[0] if len(h) > 2 else np.nan, t / 3600.0]); starts.append(t)
    F = pd.DataFrame(rows, columns=["acc_std", "acc_diff", "acc_p95", "acc_moving", "acc_range", "hr", "hr_std", "hr_slope", "hours"])
    F[["hr", "hr_std", "hr_slope"]] = F[["hr", "hr_std", "hr_slope"]].interpolate(limit_direction="both")
    F["hr_rel"] = F.hr - F.hr.median(); F["hr_z"] = F.hr_rel / (F.hr.std() + 1e-9)
    F["hours_frac"] = F.hours / max(F.hours.max(), 1e-9); F["cos_clock"] = np.cos(2 * np.pi * F.hours / 24)
    base = ["acc_std", "acc_diff", "acc_p95", "acc_moving", "hr_rel", "hr_z", "hr_std"]
    for w in (5, 11, 21, 41):
        R = F[base].rolling(w, center=True, min_periods=1)
        F[[f"{c}_m{w}" for c in base]] = R.mean().values; F[[f"{c}_s{w}" for c in base]] = R.std().fillna(0).values
    for k in (1, 2, 4): F[[f"acc_std_lag{k}", f"acc_std_lead{k}"]] = np.c_[F.acc_std.shift(k).bfill(), F.acc_std.shift(-k).ffill()]
    F["since_move"] = (F.acc_moving < 0.05).astype(int).groupby((F.acc_moving >= 0.05).cumsum()).cumsum()
    F = np.nan_to_num(F.drop(columns=["hours"]).values)
    Fp = sleep_personal(F)
    return np.nan_to_num(np.hstack([F, Fp, sleep_long_ctx(Fp[:, :8])])).astype(np.float32), np.array(starts)

def sleep_summary(hyp, epoch_s=30):
    hyp = np.asarray(hyp); asleep = hyp != "W"
    if not asleep.any(): return dict(TST_h=0.0, efficiency=0.0)
    on, off = int(np.argmax(asleep)), len(asleep) - int(np.argmax(asleep[::-1]))
    tst = asleep.sum() * epoch_s / 60; seg = hyp[on:off]
    return dict(TST_h=round(tst / 60, 2), efficiency=round(100 * tst / (len(hyp) * epoch_s / 60), 1),
                latency_min=round(on * epoch_s / 60, 1), WASO_min=round((seg == "W").sum() * epoch_s / 60, 1),
                awakenings=int(np.sum((seg[1:] == "W") & (seg[:-1] != "W"))),
                deep_min=round(np.isin(hyp, ["Deep", "N3"]).sum() * epoch_s / 60, 1), REM_min=round((hyp == "REM").sum() * epoch_s / 60, 1))

def imu_windows_har(acc_xyz_g, gyro_xyz, fs=50):
    """Режим невесомости (UCI HAR): окна 2.56 с с шагом 1.28 с; ускорение без гравитации + гироскоп, только модули."""
    acc = highpass(np.asarray(acc_xyz_g, float), fs); gyr = np.asarray(gyro_xyz, float); W, S = int(2.56 * fs), int(1.28 * fs)
    rows, starts = [], []
    for st in range(0, len(acc) - W + 1, S):
        rows.append(mag_feats(np.linalg.norm(acc[st:st+W], axis=1), fs) + mag_feats(np.linalg.norm(gyr[st:st+W], axis=1), fs)); starts.append(st / fs)
    return np.nan_to_num(np.array(rows)), np.array(starts)

def imu_windows_training(wrist_acc, wrist_gyro, chest_acc, chest_gyro, hr, fs=100):
    """Тренировки (PAMAP2): окна 5.12 с, шаг 2.56 с; ускорение м/с² (без гравитации), гироскоп рад/с, пульс уд/мин."""
    wa, ca = highpass(np.asarray(wrist_acc, float), fs), highpass(np.asarray(chest_acc, float), fs)
    wg, cg, hr = np.asarray(wrist_gyro, float), np.asarray(chest_gyro, float), np.asarray(hr, float); W, S = int(5.12 * fs), int(2.56 * fs)
    rows, starts = [], []
    for st in range(0, len(wa) - W + 1, S):
        sl = slice(st, st + W)
        fw = mag_feats(np.linalg.norm(wa[sl], axis=1), fs) + mag_feats(np.linalg.norm(wg[sl], axis=1), fs)
        fc = mag_feats(np.linalg.norm(ca[sl], axis=1), fs) + mag_feats(np.linalg.norm(cg[sl], axis=1), fs)
        rows.append(fw + fc + [float(np.nanmean(hr[sl]))]); starts.append(st / fs)
    return np.nan_to_num(np.array(rows)), np.array(starts)

def radiation_analyze(t, rate, lat, lon, L, baseline_days=7.0, zthr=6.0, persist=2, mission_days=180):
    """Мощность поглощённой дозы (мкГр/ч) по орбите. Норма для каждого места орбиты (L, широта, долгота) — по первым
    baseline_days суток; тревога — превышение в высоких широтах (L > 3) z > zthr два 3-часовых окна подряд."""
    d = pd.DataFrame(dict(t=pd.to_datetime(t, utc=True), rate=rate, latitude=lat, longitude=lon, l=L)).dropna().sort_values("t")
    dt = d.t.diff().dt.total_seconds().fillna(0).clip(upper=3600).values / 3600
    cum_mGy = float(np.sum(0.5 * (d.rate.values[1:] + d.rate.values[:-1]) * dt[1:]) / 1000)
    days = max((d.t.iloc[-1] - d.t.iloc[0]).total_seconds() / 86400, 1e-6)
    res = dict(accumulated_mGy=round(cum_mGy, 4), days=round(days, 2), daily_mGy=round(cum_mGy / days, 4),
               projected_mission_mGy=round(cum_mGy / days * mission_days, 2), alarms=[])
    q1 = d.t.iloc[0] + pd.Timedelta(days=baseline_days); quiet = d.t < q1
    if quiet.sum() < 500: res["note"] = "мало данных для орбитальной нормы (нужно ≥ baseline_days суток)"; return res
    d["Lb"] = pd.cut(d["l"].clip(1, 8), np.linspace(1, 8, 29)); d["lonb"] = pd.cut(d["longitude"], np.linspace(-180, 180, 19))
    d["latb"] = pd.cut(d["latitude"], np.linspace(-55, 55, 12)); key = ["Lb", "lonb", "latb"]
    d = d.join(d[quiet].groupby(key, observed=True)["rate"].median().rename("exp"), on=key); hi = (d["l"] > 3).values
    h = pd.Series((d.rate / d.exp).values[hi], index=pd.DatetimeIndex(d.t[hi])).resample("3h").median()
    z = (h - 1) / (h[h.index < q1].std() + 1e-9)
    alarm = (z > zthr).astype(int).rolling(persist, min_periods=persist).sum().eq(persist) & (h.index >= q1)
    res["alarms"] = [dict(window_start=str(i), ratio_to_orbit_norm=round(float(h[i]), 2), z=round(float(z[i]), 1)) for i in h.index[alarm]]
    res["max_ratio_to_orbit_norm"] = round(float(h[h.index >= q1].max()), 2) if (h.index >= q1).any() else None
    return res
