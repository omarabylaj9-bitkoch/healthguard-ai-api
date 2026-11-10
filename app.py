"""HealthGuard AI — API. Не ставит диагнозов: только замечает отклонения от личной нормы и уведомляет.
Документация с примерами: /docs"""
import os, base64, numpy as np, pandas as pd, joblib
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
        self[k] = joblib.load(p); return self[k]
    def get(self, k, default=None):
        if k in self.settings: return self.settings[k]
        try: return self[k]
        except KeyError: return default
M = _Models()
app = FastAPI(title="HealthGuard AI API", version="1.0",
              description="Мониторинг здоровья астронавтов: Heart, Sleep, Activity, Mental, Radiation и Unified Health Engine. "
                          "Система **не ставит диагнозов** — она сообщает об отклонениях от личной нормы человека.")

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
class HeartIn(BaseModel):
    ecg: List[float] = Field(..., description="ЭКГ, 1-е отведение (мВ). Не меньше ~5 минут: первые 300 сокращений — личная калибровка")
    fs: float = Field(360, description="Частота дискретизации, Гц")
    ecg_lead2: Optional[List[float]] = Field(None, description="2-е отведение (необязательно) — для оценки качества v2")
    return_beats: bool = Field(False, description="Вернуть тип каждого сокращения")

@app.post("/heart")
def heart(inp: HeartIn):
    sig, fs = F.to_360(inp.ecg, inp.fs)
    if len(sig) < 60 * fs: raise HTTPException(400, "нужно хотя бы 60 с ЭКГ (лучше ≥ 5 мин)")
    beats = F.detect_beats(sig, fs)
    if len(beats) < 30: raise HTTPException(422, "слишком мало сокращений найдено — проверьте сигнал")
    rr = np.diff(beats) / fs; q = F.ecg_quality(sig, fs, beats); good = np.mean([x[2] for x in q]) if q else 0.0
    if inp.ecg_lead2 is not None and M.get("heart_ecg_quality") is not None:
        s2, _ = F.to_360(inp.ecg_lead2, inp.fs); n = min(len(sig), len(s2)); d1 = F.detect_beats(s2[:n], fs)
        Fq = [F.sqi_feats(sig[:n], s2[:n], fs, beats, d1, a, b) for a, b, _ in q if b <= n]
        good = float(np.mean(M["heart_ecg_quality"].predict_proba(np.array(Fq))[:, 1] < 0.5)) if Fq else good
    X, _ = F.beat_features(sig, beats, ["N"] * len(beats), fs)
    if len(X) == 0: raise HTTPException(422, "не удалось выделить сокращения")
    if M.get("heart_personal_norm", True): X = F.personal_norm(X)
    P = _proba(M["heart"], X); lab = P.columns[P.values.argmax(1)]
    bt = beats[10:10 + len(lab)]; isn = np.asarray(lab) == "N"            # RR только между двумя нормальными (N–N), как в стандарте HRV
    nn = (np.diff(bt) / fs)[isn[:-1] & isn[1:]] if len(bt) > 2 else rr; nn = nn[(nn > 0.3) & (nn < 2.0)]
    out = dict(beats=int(len(beats)), heart_rate_bpm=round(60 / float(np.median(nn)), 1), rmssd_ms=round(float(np.sqrt(np.mean(np.diff(nn * 1000) ** 2))), 1),
               ectopic_pct=round(100 * float(np.mean(lab != "N")), 2),
               beat_types={k: int(v) for k, v in pd.Series(lab).value_counts().items()},
               good_signal_fraction=round(float(good), 3),
               notice="Автоматический режим (без разметки врача). Доля внеочередных > 1% — повод показать ЭКГ врачу.")
    if inp.return_beats: out["beats_detail"] = [dict(t_s=round(float(beats[i + 10]) / fs, 3), type=str(l)) for i, l in enumerate(lab) if i + 10 < len(beats)]
    return out

# ---------------- Sleep ----------------
class HeadbandIn(BaseModel):
    eeg_fpz_cz: Optional[List[float]] = Field(None, description="Лобный канал ЭЭГ повязки (Fpz-Cz). Для целой ночи лучше eeg_fpz_cz_b64")
    eog: Optional[List[float]] = Field(None, description="Канал движений глаз (лобные электроды повязки), та же длина")
    eeg_fpz_cz_b64: Optional[str] = Field(None, description="То же в компактном виде: float32 little-endian, base64 (для целой ночи)")
    eog_b64: Optional[str] = Field(None, description="EOG в виде float32 little-endian, base64")
    fs: float = Field(100, description="Частота дискретизации, Гц")
    units: str = Field("uV", description="Единицы сигналов: 'uV' или 'V'")

@app.post("/sleep/headband")
def sleep_headband(inp: HeadbandIn):
    k = 1e-6 if inp.units.lower() == "uv" else 1.0
    def arr(lst, b64, name):
        if b64: return np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(float)
        if lst is None: raise HTTPException(400, f"нет данных {name}")
        return np.asarray(lst, float)
    eeg, eog = arr(inp.eeg_fpz_cz, inp.eeg_fpz_cz_b64, "eeg_fpz_cz") * k, arr(inp.eog, inp.eog_b64, "eog") * k
    if len(eeg) < 30 * inp.fs * 20: raise HTTPException(400, "нужно хотя бы 10 минут записи")
    F23 = F.headband_epoch_features(eeg, eog, inp.fs); X = F.headband_model_input(F23)
    P = _proba(M["sleep_headband"], X); hyp = P.columns[P.values.argmax(1)]
    return dict(epochs=int(len(hyp)), epoch_s=30, hypnogram=[str(h) for h in hyp], summary=F.sleep_summary(hyp),
                stages="W — бодрствование, Light — лёгкий сон, Deep — глубокий, REM — быстрый сон")

class WristIn(BaseModel):
    t_acc: List[float] = Field(..., description="Время отсчётов акселерометра, с от начала ночи")
    acc_xyz: List[List[float]] = Field(..., description="Ускорение браслета [[x,y,z],...], g")
    t_hr: List[float] = Field(..., description="Время отсчётов пульса, с")
    hr: List[float] = Field(..., description="Пульс, уд/мин")

@app.post("/sleep/wrist")
def sleep_wrist(inp: WristIn):
    X, starts = F.wrist_epoch_features(inp.t_acc, inp.acc_xyz, inp.t_hr, inp.hr)
    if len(X) < 20: raise HTTPException(400, "нужно хотя бы 10 минут записи")
    P = _proba(M["sleep_wrist"], X); lab = np.where(P.values.argmax(1) == list(P.columns).index("W"), "W", "Sleep")
    s = F.sleep_summary(lab)
    return dict(epochs=int(len(lab)), epoch_s=30, sleep_wake=[str(v) for v in lab],
                summary=dict(TST_h=s["TST_h"], efficiency=s.get("efficiency"), latency_min=s.get("latency_min"), WASO_min=s.get("WASO_min")),
                notice="Браслет: только сон/бодрствование, время и эффективность сна. Стадии — через /sleep/headband.")

# ---------------- Activity ----------------
class ImuIn(BaseModel):
    acc_xyz: List[List[float]] = Field(..., description="Ускорение [[x,y,z],...], g, датчик на поясе/туловище (как смартфон на поясе в UCI HAR); гравитация убирается на сервере")
    gyro_xyz: List[List[float]] = Field(..., description="Гироскоп [[x,y,z],...], рад/с")
    fs: float = Field(50, description="Частота, Гц (модель обучена на 50 Гц)")

@app.post("/activity/microgravity")
def activity_mg(inp: ImuIn):
    X, starts = F.imu_windows_har(inp.acc_xyz, inp.gyro_xyz, inp.fs)
    if len(X) < 1: raise HTTPException(400, "нужно хотя бы 2.56 с данных")
    g = np.zeros(len(X)); P = F.smooth_proba(M["activity_microgravity"].predict_proba(F.har_context(X, g)), g)
    lab = M["activity_microgravity"].classes_[P.argmax(1)]
    return dict(windows=[dict(t_s=round(float(t), 2), activity=str(a)) for t, a in zip(starts, lab)],
                share={k: round(float(v), 3) for k, v in pd.Series(lab).value_counts(normalize=True).items()})

class TrainingIn(BaseModel):
    wrist_acc: List[List[float]] = Field(..., description="Браслет: ускорение [[x,y,z]], м/с²")
    wrist_gyro: List[List[float]] = Field(..., description="Браслет: гироскоп, рад/с")
    chest_acc: List[List[float]] = Field(..., description="Грудной датчик: ускорение, м/с²")
    chest_gyro: List[List[float]] = Field(..., description="Грудной датчик: гироскоп, рад/с")
    hr: List[float] = Field(..., description="Пульс на каждый отсчёт, уд/мин")
    fs: float = Field(100, description="Частота, Гц (модель обучена на 100 Гц)")

@app.post("/activity/training")
def activity_training(inp: TrainingIn):
    X, starts = F.imu_windows_training(inp.wrist_acc, inp.wrist_gyro, inp.chest_acc, inp.chest_gyro, inp.hr, inp.fs)
    if len(X) < 1: raise HTTPException(400, "нужно хотя бы 5.12 с данных")
    g = np.zeros(len(X)); P = F.smooth_proba(M["activity_pamap2_microgravity"].predict_proba(F.har_context(X, g)), g)
    lab = M["activity_pamap2_microgravity"].classes_[P.argmax(1)]
    return dict(windows=[dict(t_s=round(float(t), 2), activity=str(a)) for t, a in zip(starts, lab)],
                minutes={k: round(float(v) * 2.56 / 60, 1) for k, v in pd.Series(lab).value_counts().items()})

# ---------------- Mental ----------------
class MentalIn(BaseModel):
    chest_ecg: List[float]; chest_eda: List[float]; chest_resp: List[float]; chest_temp: List[float]
    chest_acc: List[List[float]] = Field(..., description="Грудной датчик 700 Гц: ЭКГ (мВ), EDA (мкСм), дыхание (%), температура (°C), ускорение (g)")
    wrist_bvp: List[float] = Field(..., description="Браслет: пульсовая волна, 64 Гц")
    wrist_eda: List[float] = Field(..., description="Браслет: EDA, мкСм, 4 Гц")
    wrist_temp: List[float] = Field(..., description="Браслет: температура кожи, °C, 4 Гц")
    wrist_acc: List[List[float]] = Field(..., description="Браслет: ускорение, 32 Гц")

@app.post("/mental")
def mental(inp: MentalIn):
    """Первые 10 минут записи должны быть спокойными — это личная норма человека."""
    n = len(inp.chest_ecg)
    if n < 700 * 60 * 12: raise HTTPException(400, "нужно ≥ 12 минут записи: первые 10 минут — спокойная личная норма")
    d = {"label": np.ones(n), "signal": {"chest": {"ECG": np.asarray(inp.chest_ecg)[:, None], "EDA": np.asarray(inp.chest_eda)[:, None],
         "Resp": np.asarray(inp.chest_resp)[:, None], "Temp": np.asarray(inp.chest_temp)[:, None], "ACC": np.asarray(inp.chest_acc)},
         "wrist": {"BVP": np.asarray(inp.wrist_bvp)[:, None], "EDA": np.asarray(inp.wrist_eda)[:, None], "TEMP": np.asarray(inp.wrist_temp)[:, None],
                   "ACC": np.asarray(inp.wrist_acc)}}}
    xc, xw, _ = F.wesad_windows(d)
    X = F.personal_start(np.hstack([xc, xw])); p = M["mental"].predict_proba(X)[:, list(M["mental"].classes_).index(1)]
    return dict(windows_60s_step_15s=int(len(p)), stress_probability=[round(float(v), 3) for v in p],
                stress_fraction=round(float(np.mean(p >= 0.5)), 3),
                notice="Это не диагноз: физиологические признаки стрессовой реакции относительно личной нормы.")

# ---------------- Radiation ----------------
class RadIn(BaseModel):
    timestamp: List[str] = Field(..., description="Время ISO 8601 (UTC)")
    absorbed_dose_rate: List[float] = Field(..., description="Мощность поглощённой дозы, мкГр/ч")
    latitude: List[float]; longitude: List[float]
    L: List[float] = Field(..., description="L-оболочка (как в NASA RadLab)")
    baseline_days: float = Field(7.0, description="Сколько первых суток считать спокойной орбитальной нормой")
    mission_days: int = Field(180)

@app.post("/radiation")
def radiation(inp: RadIn):
    return F.radiation_analyze(inp.timestamp, inp.absorbed_dose_rate, inp.latitude, inp.longitude, inp.L,
                               inp.baseline_days, mission_days=inp.mission_days)

# ---------------- Unified Health Engine ----------------
class EngineIn(BaseModel):
    days: List[Dict[str, Optional[float]]] = Field(..., description="По дню на элемент: resting_hr, rmssd, ectopic_pct, sleep_h, sleep_eff, "
                                                   "active_min, rad_mSv, stress_frac, pvt_rt. Первые 14 дней — личная норма.")
    radiation_event_days: List[int] = Field(default_factory=list, description="Номера дней с радиационным событием (из /radiation)")

@app.post("/engine")
def engine(inp: EngineIn):
    df = pd.DataFrame(inp.days).astype(float); df.index = range(len(df))
    if len(df) < 15: raise HTTPException(400, "нужно ≥ 15 дней: первые 14 — личная норма")
    eng = F.UnifiedHealthEngine().fit_baseline(df); lvl = eng.assess(df, rad_events_days=inp.radiation_event_days)
    out = []
    for d in df.index:
        e = eng.explain(df, d); e["notification"] = eng.notify(df, d); out.append(e)
    return dict(levels=[F.LEVELS[int(v)] for v in lvl], days=out)
