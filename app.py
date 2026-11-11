"""HealthGuard AI API. Not a diagnostic tool: it detects deviations from a person's baseline and notifies. Docs: /docs"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import base64, numpy as np, pandas as pd, joblib
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import features as F

class _Models(dict):
    """Модели грузятся по требованию (по одной на модуль) — экономим память на бесплатном хостинге (512 МБ)."""
    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    def __init__(self):
        super().__init__(); self.settings = joblib.load(os.path.join(self.DIR, "settings.joblib"))
    def __missing__(self, k):
        p = os.path.join(self.DIR, k + ".joblib")
        if not os.path.exists(p): raise KeyError(k)
        m = joblib.load(p)
        for est in [m] + [st for _, st in getattr(m, "steps", [])]:   # на 0.1 ядра параллельность только мешает
            if hasattr(est, "n_jobs"): est.n_jobs = 1
        self[k] = m; return m
    def get(self, k, default=None):
        if k in self.settings: return self.settings[k]
        try: return self[k]
        except KeyError: return default
M = _Models()
app = FastAPI(title="HealthGuard AI API", version="1.0",
              description="Astronaut health monitoring: Heart, Sleep, Activity, Mental, Radiation and the Unified Health Engine. "
                          "The system **does not diagnose**: it reports deviations from each person's own baseline.")

def _proba(model, X):
    return pd.DataFrame(model.predict_proba(X), columns=model.classes_)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "HealthGuard AI API", "docs": "/docs", "modules": ["/heart", "/sleep/headband", "/sleep/wrist", "/activity/microgravity",
            "/activity/training", "/mental", "/radiation", "/engine"]}

# ---------------- Heart ----------------
def _arr(lst, b64, name, required=True):
    """Сигнал списком чисел или компактно: float32 little-endian в base64 (в разы быстрее для длинных записей)."""
    if b64: return np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(float)
    if lst is None:
        if required: raise HTTPException(400, f"missing data: {name}")
        return None
    return np.asarray(lst, float)

class HeartIn(BaseModel):
    ecg: Optional[List[float]] = Field(None, description="ECG lead 1 (mV). At least ~5 min: the first 300 beats are used for personal calibration")
    ecg_b64: Optional[str] = Field(None, description="Same, compact: float32 little-endian, base64 (recommended, faster)")
    ecg_lead2_b64: Optional[str] = Field(None, description="Lead 2, compact (optional)")
    fs: float = Field(360, description="Sampling rate, Hz")
    ecg_lead2: Optional[List[float]] = Field(None, description="Lead 2 (optional), used for signal quality v2")
    return_beats: bool = Field(False, description="Return the type of every beat")

@app.post("/heart")
def heart(inp: HeartIn):
    sig, fs = F.to_360(_arr(inp.ecg, inp.ecg_b64, "ecg"), inp.fs)
    lead2 = _arr(inp.ecg_lead2, inp.ecg_lead2_b64, "ecg_lead2", required=False)
    if len(sig) < 60 * fs: raise HTTPException(400, "at least 60 s of ECG required (5+ min recommended)")
    beats = F.detect_beats(sig, fs)
    if len(beats) < 30: raise HTTPException(422, "too few beats detected, check the signal")
    rr = np.diff(beats) / fs; q = F.ecg_quality(sig, fs, beats); good = np.mean([x[2] for x in q]) if q else 0.0
    if lead2 is not None and M.get("heart_ecg_quality") is not None:
        s2, _ = F.to_360(lead2, inp.fs); n = min(len(sig), len(s2)); d1 = F.detect_beats(s2[:n], fs)
        Fq = [F.sqi_feats(sig[:n], s2[:n], fs, beats, d1, a, b) for a, b, _ in q if b <= n]
        good = float(np.mean(M["heart_ecg_quality"].predict_proba(np.array(Fq))[:, 1] < 0.5)) if Fq else good
    X, _ = F.beat_features(sig, beats, ["N"] * len(beats), fs)
    if len(X) == 0: raise HTTPException(422, "could not extract beats")
    if M.get("heart_personal_norm", True): X = F.personal_norm(X)
    P = _proba(M["heart"], X); lab = P.columns[P.values.argmax(1)]
    bt = beats[10:10 + len(lab)]; isn = np.asarray(lab) == "N"            # RR только между двумя нормальными (N–N), как в стандарте HRV
    nn = (np.diff(bt) / fs)[isn[:-1] & isn[1:]] if len(bt) > 2 else rr; nn = nn[(nn > 0.3) & (nn < 2.0)]
    if len(nn) < 3: nn = rr[(rr > 0.3) & (rr < 2.0)]                       # почти нет нормальных подряд — берём все интервалы
    out = dict(beats=int(len(beats)), heart_rate_bpm=round(60 / float(np.median(nn)), 1), rmssd_ms=round(float(np.sqrt(np.mean(np.diff(nn * 1000) ** 2))), 1),
               ectopic_pct=round(100 * float(np.mean(lab != "N")), 2),
               beat_types={k: int(v) for k, v in pd.Series(lab).value_counts().items()},
               good_signal_fraction=round(float(good), 3),
               notice="Automatic mode (no physician labels). Ectopic share above 1% is a reason to show the ECG to a physician.")
    if inp.return_beats: out["beats_detail"] = [dict(t_s=round(float(beats[i + 10]) / fs, 3), type=str(l)) for i, l in enumerate(lab) if i + 10 < len(beats)]
    return out

# ---------------- Sleep ----------------
class HeadbandIn(BaseModel):
    eeg_fpz_cz: Optional[List[float]] = Field(None, description="Headband frontal EEG channel (Fpz-Cz). For a full night use eeg_fpz_cz_b64")
    eog: Optional[List[float]] = Field(None, description="Eye movement channel (headband frontal electrodes), same length")
    eeg_fpz_cz_b64: Optional[str] = Field(None, description="Same, compact: float32 little-endian, base64 (for a full night)")
    eog_b64: Optional[str] = Field(None, description="EOG as float32 little-endian, base64")
    fs: float = Field(100, description="Sampling rate, Hz")
    units: str = Field("uV", description="Signal units: 'uV' or 'V'")

@app.post("/sleep/headband")
def sleep_headband(inp: HeadbandIn):
    k = 1e-6 if inp.units.lower() == "uv" else 1.0
    def arr(lst, b64, name):
        if b64: return np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(float)
        if lst is None: raise HTTPException(400, f"missing data: {name}")
        return np.asarray(lst, float)
    eeg, eog = arr(inp.eeg_fpz_cz, inp.eeg_fpz_cz_b64, "eeg_fpz_cz") * k, arr(inp.eog, inp.eog_b64, "eog") * k
    if len(eeg) < 30 * inp.fs * 20: raise HTTPException(400, "at least 10 minutes of recording required")
    F23 = F.headband_epoch_features(eeg, eog, inp.fs); X = F.headband_model_input(F23)
    P = _proba(M["sleep_headband"], X); hyp = P.columns[P.values.argmax(1)]
    return dict(epochs=int(len(hyp)), epoch_s=30, hypnogram=[str(h) for h in hyp], summary=F.sleep_summary(hyp),
                stages="W = wake, Light = light sleep, Deep = deep sleep, REM = REM sleep")

class WristIn(BaseModel):
    t_acc: List[float] = Field(..., description="Accelerometer timestamps, s from start of night")
    acc_xyz: List[List[float]] = Field(..., description="Wrist acceleration [[x,y,z],...], g")
    t_hr: List[float] = Field(..., description="Heart rate timestamps, s")
    hr: List[float] = Field(..., description="Heart rate, bpm")

@app.post("/sleep/wrist")
def sleep_wrist(inp: WristIn):
    X, starts = F.wrist_epoch_features(inp.t_acc, inp.acc_xyz, inp.t_hr, inp.hr)
    if len(X) < 20: raise HTTPException(400, "at least 10 minutes of recording required")
    P = _proba(M["sleep_wrist"], X); lab = np.where(P.values.argmax(1) == list(P.columns).index("W"), "W", "Sleep")
    s = F.sleep_summary(lab)
    return dict(epochs=int(len(lab)), epoch_s=30, sleep_wake=[str(v) for v in lab],
                summary=dict(TST_h=s["TST_h"], efficiency=s.get("efficiency"), latency_min=s.get("latency_min"), WASO_min=s.get("WASO_min")),
                notice="Wrist mode: sleep/wake, total sleep time and efficiency only. For stages use /sleep/headband.")

# ---------------- Activity ----------------
class ImuIn(BaseModel):
    acc_xyz: List[List[float]] = Field(..., description="Acceleration [[x,y,z],...], g, waist/trunk sensor (like the waist phone in UCI HAR); gravity is removed on the server")
    gyro_xyz: List[List[float]] = Field(..., description="Gyroscope [[x,y,z],...], rad/s")
    fs: float = Field(50, description="Sampling rate, Hz (model trained at 50 Hz)")

@app.post("/activity/microgravity")
def activity_mg(inp: ImuIn):
    X, starts = F.imu_windows_har(inp.acc_xyz, inp.gyro_xyz, inp.fs)
    if len(X) < 1: raise HTTPException(400, "at least 2.56 s of data required")
    g = np.zeros(len(X)); P = F.smooth_proba(M["activity_microgravity"].predict_proba(F.har_context(X, g)), g)
    lab = M["activity_microgravity"].classes_[P.argmax(1)]
    return dict(windows=[dict(t_s=round(float(t), 2), activity=str(a)) for t, a in zip(starts, lab)],
                share={k: round(float(v), 3) for k, v in pd.Series(lab).value_counts(normalize=True).items()})

class TrainingIn(BaseModel):
    wrist_acc: List[List[float]] = Field(..., description="Wrist: acceleration [[x,y,z]], m/s²")
    wrist_gyro: List[List[float]] = Field(..., description="Wrist: gyroscope, rad/s")
    chest_acc: List[List[float]] = Field(..., description="Chest sensor: acceleration, m/s²")
    chest_gyro: List[List[float]] = Field(..., description="Chest sensor: gyroscope, rad/s")
    hr: List[float] = Field(..., description="Heart rate for every sample, bpm")
    fs: float = Field(100, description="Sampling rate, Hz (model trained at 100 Hz)")

@app.post("/activity/training")
def activity_training(inp: TrainingIn):
    X, starts = F.imu_windows_training(inp.wrist_acc, inp.wrist_gyro, inp.chest_acc, inp.chest_gyro, inp.hr, inp.fs)
    if len(X) < 1: raise HTTPException(400, "at least 5.12 s of data required")
    g = np.zeros(len(X)); P = F.smooth_proba(M["activity_pamap2_microgravity"].predict_proba(F.har_context(X, g)), g)
    lab = M["activity_pamap2_microgravity"].classes_[P.argmax(1)]
    return dict(windows=[dict(t_s=round(float(t), 2), activity=str(a)) for t, a in zip(starts, lab)],
                minutes={k: round(float(v) * 2.56 / 60, 1) for k, v in pd.Series(lab).value_counts().items()})

# ---------------- Mental ----------------
class MentalIn(BaseModel):
    chest_ecg: List[float]; chest_eda: List[float]; chest_resp: List[float]; chest_temp: List[float]
    chest_acc: List[List[float]] = Field(..., description="Chest sensor, 700 Hz: ECG (mV), EDA (µS), respiration (%), temperature (°C), acceleration (g)")
    wrist_bvp: List[float] = Field(..., description="Wrist: blood volume pulse, 64 Hz")
    wrist_eda: List[float] = Field(..., description="Wrist: EDA, µS, 4 Hz")
    wrist_temp: List[float] = Field(..., description="Wrist: skin temperature, °C, 4 Hz")
    wrist_acc: List[List[float]] = Field(..., description="Wrist: acceleration, 32 Hz")

@app.post("/mental")
def mental(inp: MentalIn):
    """The first 10 minutes must be calm: they form the personal baseline."""
    n = len(inp.chest_ecg)
    if n < 700 * 60 * 12: raise HTTPException(400, "at least 12 minutes required: the first 10 minutes are the calm baseline")
    d = {"label": np.ones(n), "signal": {"chest": {"ECG": np.asarray(inp.chest_ecg)[:, None], "EDA": np.asarray(inp.chest_eda)[:, None],
         "Resp": np.asarray(inp.chest_resp)[:, None], "Temp": np.asarray(inp.chest_temp)[:, None], "ACC": np.asarray(inp.chest_acc)},
         "wrist": {"BVP": np.asarray(inp.wrist_bvp)[:, None], "EDA": np.asarray(inp.wrist_eda)[:, None], "TEMP": np.asarray(inp.wrist_temp)[:, None],
                   "ACC": np.asarray(inp.wrist_acc)}}}
    xc, xw, _ = F.wesad_windows(d)
    X = F.personal_start(np.hstack([xc, xw])); p = M["mental"].predict_proba(X)[:, list(M["mental"].classes_).index(1)]
    return dict(windows_60s_step_15s=int(len(p)), stress_probability=[round(float(v), 3) for v in p],
                stress_fraction=round(float(np.mean(p >= 0.5)), 3),
                notice="Not a diagnosis: physiological signs of a stress response relative to the personal baseline.")

# ---------------- Radiation ----------------
class RadIn(BaseModel):
    timestamp: List[str] = Field(..., description="Time, ISO 8601 (UTC)")
    absorbed_dose_rate: List[float] = Field(..., description="Absorbed dose rate, µGy/h")
    latitude: List[float]; longitude: List[float]
    L: List[float] = Field(..., description="L-shell (as in NASA RadLab)")
    baseline_days: float = Field(7.0, description="How many initial days form the quiet orbital baseline")
    mission_days: int = Field(180)

@app.post("/radiation")
def radiation(inp: RadIn):
    return F.radiation_analyze(inp.timestamp, inp.absorbed_dose_rate, inp.latitude, inp.longitude, inp.L,
                               inp.baseline_days, mission_days=inp.mission_days)

# ---------------- Unified Health Engine ----------------
class EngineIn(BaseModel):
    days: List[Dict[str, Optional[float]]] = Field(..., description="One item per day: resting_hr, rmssd, ectopic_pct, sleep_h, sleep_eff, "
                                                   "active_min, rad_mSv, stress_frac, pvt_rt. The first 14 days form the personal baseline.")
    radiation_event_days: List[int] = Field(default_factory=list, description="Day indices with a radiation event (from /radiation)")
    lang: str = Field("en", description="Notification language: 'en' (default) or 'ru'")

@app.post("/engine")
def engine(inp: EngineIn):
    df = pd.DataFrame(inp.days).astype(float); df.index = range(len(df))
    if len(df) < 15: raise HTTPException(400, "at least 15 days required: the first 14 form the baseline")
    eng = F.UnifiedHealthEngine().fit_baseline(df); lvl = eng.assess(df, rad_events_days=inp.radiation_event_days)
    out = []
    for d in df.index:
        e = eng.explain(df, d, inp.lang); e["notification"] = eng.notify(df, d, inp.lang); out.append(e)
    return dict(levels=[F.LEVELS[int(v)] for v in lvl], days=out)
