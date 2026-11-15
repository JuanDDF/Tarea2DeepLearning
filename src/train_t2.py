# -*- coding: utf-8 -*-
"""Tarea 2 (sección 2.21): búsqueda de arquitecturas MLP para BTCUSDT diario.

Reanudable: cada candidato terminado se guarda en Drive (JSONL con escritura atómica).
Al relanzar, lo ya hecho se salta. Topes de tiempo, medidos desde T0 (guardado en Drive):
  - T0 + 16 h 45: la búsqueda de las 15 configuraciones restantes (F4) no inicia más candidatos.
  - T0 + 17 h 45: ninguna fase inicia tareas nuevas.
  - T0 + 18 h 00: el proceso y sus hijos se matan (temporizador interno + vigilante externo).
Fases:
  F0 particiones tsxv (caché) -> F1 búsqueda completa en la configuración representativa ->
  F2 ganadora de F1 evaluada en las 16 configuraciones con las 3 estrategias ->
  F3 modelo final + hold-out con la ganadora de F1 (entregable mínimo completo) ->
  F4 búsqueda completa en las otras 15 configuraciones ->
  F5 ganadora propia por configuración: modelo final, Top-5 en hold-out, re-evaluación fc/kf ->
  F6 detalle por fold (16 config x 3 estrategias x 5 folds): predicciones de train y val en USD,
     métricas por paso de horizonte, p-valor BDS (h=1) y baselines MLPRegressor e ingenuo.
"""
import os, sys, json, time, glob, random, hashlib, threading, signal, traceback, itertools, gc, shutil, uuid
import numpy as np, pandas as pd
import multiprocessing as mp

ROOT = os.environ.get("T2_ROOT", "/content/drive/MyDrive/tarea2_series")
SMOKE = os.environ.get("T2_SMOKE") == "1"
OUT = f"{ROOT}/results_smoke" if SMOKE else f"{ROOT}/results"
RAW, ERR, MODELS = f"{OUT}/raw", f"{OUT}/errors", f"{OUT}/models"
CACHE = f"{ROOT}/cache"
T0_FILE = f"{OUT}/T0.txt"
for _d in (RAW, ERR, MODELS, CACHE):
    os.makedirs(_d, exist_ok=True)

# ---------------- configuración (no cambiar con el entrenamiento en marcha) ----------------
LAGS, HS = [7, 14, 21, 28], [7, 14, 21, 28]
REP = (14, 14)                                   # configuración representativa (lag, H)
F4_ORDER = [(7, 7), (28, 28), (21, 21), (7, 14), (14, 7), (28, 14), (14, 28), (7, 21),
            (21, 7), (28, 21), (21, 28), (7, 28), (28, 7), (14, 21), (21, 14)]
if SMOKE:
    F4_ORDER = [(7, 7)]
ALL_CONFIGS = [REP] + F4_ORDER
NEURONAS, MAX_CAPAS = [32, 64, 128, 256], 3
EPOCHS = 2 if SMOKE else 30
BATCH, DROP = 32, 0.5
JUMP = 1            # kFold de tsxv no termina si jump > lag: jump=1 es seguro para todos los lags
N_FOLDS = 5         # folds usados por estrategia (gk tiene exactamente 5)
USE_LOG = True      # se modela log(precio); las métricas del hold-out se calculan en USD
N_WORKERS = 8
H_KILL, H_STOP_NEW, H_STOP_SEARCH = 18.0, 17.75, 16.75
CAP_FOLD_S = {"gk": 60, "fc": 150, "kf": 150, "final": 300}   # ~7x lo medido con 8 procesos
FLUSH_S = 60
GAP_MAX = 0.15


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def atomic_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as h:
        h.write(text); h.flush(); os.fsync(h.fileno())
    os.replace(tmp, path)


def series():
    y = pd.read_csv(f"{ROOT}/data/btc_1d.csv")["close"].values.astype("float64")
    n_hold = max(int(len(y) * 0.10), 56)
    z = np.log(y) if USE_LOG else y.copy()
    return y, z, n_hold


def all_candidates():
    c = []
    for n in range(1, MAX_CAPAS + 1):
        for arch in itertools.product(NEURONAS, repeat=n):
            for mask in itertools.product([False, True], repeat=n):
                c.append((arch, mask))
    return c                                        # 8 + 64 + 512 = 584


def shuffled(L, H):
    c = all_candidates()
    random.Random(f"{L}-{H}").shuffle(c)            # orden aleatorio fijo: un corte deja una muestra no sesgada
    return c[:6] if SMOKE else c


# ---------------- F0: particiones tsxv (solo proceso principal) ----------------
def split_file(L, H):
    return f"{CACHE}/splits_L{L}_H{H}_j{JUMP}.npz"


def prep_splits(L, H):
    f = split_file(L, H)
    if os.path.exists(f):
        return
    from tsxv import splitTrainVal as S
    FN = {"gk": S.split_train_val_groupKFold,
          "fc": S.split_train_val_forwardChaining,
          "kf": S.split_train_val_kFold}
    _, z, n_hold = series()
    zc = z[:-n_hold]                                # el hold-out nunca entra en la CV
    out = {}
    for s, fn in FN.items():
        t = time.time()
        X, Y, Xv, Yv = fn(zc, L, H, JUMP)
        keys = sorted(X.keys()); n = len(keys)
        if n == 0:
            raise RuntimeError(f"{s} L{L} H{H}: tsxv no devolvió folds")
        if s == "gk":
            idx = list(range(min(N_FOLDS, n)))
        elif s == "fc":
            idx = np.linspace(n // 2, n - 1, N_FOLDS)   # 5 folds repartidos en la segunda mitad
        else:
            idx = np.linspace(0, n - 1, N_FOLDS)        # 5 folds repartidos en toda la serie
        idx = sorted({int(round(i)) for i in idx})
        for j, i in enumerate(idx):
            k = keys[i]
            for nm, d, cols in (("X", X, L), ("Y", Y, H), ("Xv", Xv, L), ("Yv", Yv, H)):
                a = np.asarray(d[k], dtype="float32")
                if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != cols:
                    raise RuntimeError(f"{s} L{L} H{H} fold {k} {nm}: forma inesperada {a.shape}")
                out[f"{s}_{j}_{nm}"] = a
        out[f"{s}_nfolds_total"] = np.array(n)
        out[f"{s}_picked"] = np.array([int(keys[i]) for i in idx])
        log(f"F0 L{L} H{H} {s}: {n} folds totales, usados {[int(keys[i]) for i in idx]}, "
            f"train fold0={out[f'{s}_0_X'].shape}, val fold0={out[f'{s}_0_Xv'].shape}, {time.time()-t:.1f}s")
        del X, Y, Xv, Yv; gc.collect()
    tmp = f[:-4] + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, f)


# ---------------- trabajo de cada proceso hijo ----------------
_SPL = {}


def _init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel("ERROR")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _splits(L, H):
    if (L, H) not in _SPL:
        with np.load(split_file(L, H)) as d:
            _SPL[(L, H)] = {n: d[n] for n in d.files}
    return _SPL[(L, H)]


def _build(arch, mask, L, H):
    from tensorflow import keras
    m = keras.Sequential([keras.layers.Input((L,))])
    for u, d in zip(arch, mask):
        m.add(keras.layers.Dense(int(u), activation="relu"))
        if d:
            m.add(keras.layers.Dropout(DROP))
    m.add(keras.layers.Dense(H))
    m.compile(optimizer="adam", loss="mse",
              metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def _fit(m, X, Y, Xv, Yv, cap_s):
    from tensorflow import keras

    class Cap(keras.callbacks.Callback):          # tope por entrenamiento: corta al final de la época
        def __init__(self):
            super().__init__(); self.t = time.time(); self.hit = False

        def on_epoch_end(self, epoch, logs=None):
            if time.time() - self.t > cap_s:
                self.hit = True; self.model.stop_training = True

    cb = Cap()
    h = m.fit(X, Y, validation_data=(Xv, Yv) if Xv is not None else None, epochs=EPOCHS,
              batch_size=BATCH, shuffle=False, verbose=0, callbacks=[cb])
    return h.history, cb.hit


def _task_cv(t):
    from tensorflow import keras
    L, H, s = t["L"], t["H"], t["strat"]
    d = _splits(L, H)
    nf = sum(1 for k in d if k.startswith(f"{s}_") and k.endswith("_X") and k[len(s) + 1:-2].isdigit())
    tr, va, trunc, t0 = [], [], False, time.time()
    for j in range(nf):
        X, Y, Xv, Yv = (d[f"{s}_{j}_{n}"] for n in ("X", "Y", "Xv", "Yv"))
        mu, sd = float(X.mean()), float(X.std()) or 1.0      # scaler ajustado solo con el train del fold
        keras.backend.clear_session()
        keras.utils.set_random_seed(t["seed"] + j)
        m = _build(t["arch"], t["mask"], L, H)
        hist, hit = _fit(m, (X - mu) / sd, (Y - mu) / sd, (Xv - mu) / sd, (Yv - mu) / sd, CAP_FOLD_S[s])
        tr.append(hist["rmse"]); va.append(hist["val_rmse"]); trunc |= hit
    ne = min(len(c) for c in tr)
    trc = np.mean([c[:ne] for c in tr], axis=0)
    vac = np.mean([c[:ne] for c in va], axis=0)
    b = int(np.argmin(vac)); best = float(vac[b]); last = float(trc[-1]); gap = abs(last - best)
    return dict(kind="cv", strat=s, n_folds=nf, epochs_run=ne, truncated=bool(trunc),
                best_val_rmse=best, best_epoch=b + 1, last_train_rmse=last, gap=gap,
                gap_ratio=gap / best if best > 0 else None,
                train_curve=[float(v) for v in trc], val_curve=[float(v) for v in vac],
                val_rmse_folds=[float(min(c)) for c in va], secs=time.time() - t0)


def _windows(a, L, H, start, stop):
    """Ventanas con entrada a[i-L:i] y objetivo a[i:i+H], para i en [start, stop)."""
    X = np.stack([a[i - L:i] for i in range(start, stop)]).astype("float32")
    Y = np.stack([a[i:i + H] for i in range(start, stop)]).astype("float32")
    return X, Y


def _task_final(t):
    from tensorflow import keras
    L, H = t["L"], t["H"]
    y, z, n_hold = series()
    n = len(z); ncv = n - n_hold
    Xtr, Ytr = _windows(z, L, H, L, ncv - H + 1)        # objetivos dentro de la parte previa al hold-out
    Xho, Yho = _windows(z, L, H, ncv, n - H + 1)        # objetivos dentro del hold-out
    mu, sd = float(Xtr.mean()), float(Xtr.std())
    keras.backend.clear_session()
    keras.utils.set_random_seed(t["seed"])
    m = _build(t["arch"], t["mask"], L, H)
    t0 = time.time()
    hist, hit = _fit(m, (Xtr - mu) / sd, (Ytr - mu) / sd, None, None, CAP_FOLD_S["final"])
    P = m.predict((Xho - mu) / sd, verbose=0) * sd + mu
    inv = np.exp if USE_LOG else (lambda a: a)
    Pu, Yu = inv(P.astype("float64")), inv(Yho.astype("float64"))
    e = Pu - Yu
    r = dict(kind="final", strat="all", truncated=bool(hit), epochs_run=len(hist["rmse"]),
             train_curve=[float(v) for v in hist["rmse"]], n_train=len(Xtr), n_holdout_windows=len(Yho),
             holdout_rmse=float(np.sqrt((e ** 2).mean())), holdout_mae=float(np.abs(e).mean()),
             holdout_mse=float((e ** 2).mean()), holdout_mape=float(np.mean(np.abs(e / Yu)) * 100),
             holdout_rmse_h=[float(v) for v in np.sqrt((e ** 2).mean(axis=0))], secs=time.time() - t0)
    if t.get("save"):
        base = f"{MODELS}/{t['save']}_L{L}_H{H}"
        local = f"/tmp/{t['save']}_L{L}_H{H}_{os.getpid()}.keras"
        m.save(local)
        shutil.copyfile(local, base + ".keras.tmp"); os.replace(base + ".keras.tmp", base + ".keras")
        atomic_text(base + "_scaler.json", json.dumps(dict(mu=mu, sd=sd, use_log=USE_LOG, L=L, H=H,
                                                          arch=list(t["arch"]), mask=list(t["mask"]))))
        np.savez(base + "_preds.tmp.npz", pred=Pu, real=Yu, first_target_index=ncv)
        os.replace(base + "_preds.tmp.npz", base + "_preds.npz")
        r["saved"] = base
    return r


def _metrics(P, Y):
    """Métricas en USD por paso de horizonte (h = 1..H) y agregadas."""
    e = P - Y
    d = dict(mape_h=np.mean(np.abs(e / Y), 0) * 100, mae_h=np.mean(np.abs(e), 0),
             mse_h=np.mean(e ** 2, 0))
    d["rmse_h"] = np.sqrt(d["mse_h"])
    out = {k: [float(v) for v in a] for k, a in d.items()}
    out.update(mape=float(np.mean(np.abs(e / Y)) * 100), mae=float(np.mean(np.abs(e))),
               mse=float(np.mean(e ** 2)), rmse=float(np.sqrt(np.mean(e ** 2))))
    return out


def _bds_p(r):
    if len(r) < 20:              # con menos residuos el test BDS no es informativo
        return None
    try:
        from statsmodels.tsa.stattools import bds
        return float(np.atleast_1d(bds(np.asarray(r, dtype="float64"), max_dim=2)[1])[0])
    except Exception:
        return None


def _starts(z32, X):
    """Índice de inicio de cada ventana en la serie (tsxv no devuelve índices; se recuperan por coincidencia)."""
    pos = {}
    for i, v in enumerate(z32):
        pos.setdefault(float(v), []).append(i)
    L, out = X.shape[1], []
    for row in X:
        s = -1
        for i in pos.get(float(row[0]), []):
            if i + L <= len(z32) and np.array_equal(z32[i:i + L], row):
                s = i; break
        out.append(s)
    return np.array(out, dtype="int32")


def _task_detail(t):
    import warnings
    from tensorflow import keras
    from sklearn.neural_network import MLPRegressor
    L, H, s = t["L"], t["H"], t["strat"]
    d = _splits(L, H)
    _, z, _ = series()
    z32 = z.astype("float32")
    inv = np.exp if USE_LOG else (lambda a: a)
    nf = sum(1 for k in d if k.startswith(f"{s}_") and k.endswith("_X") and k[len(s) + 1:-2].isdigit())
    folds, arrays, t0 = [], {}, time.time()
    for j in range(nf):
        X, Y, Xv, Yv = (d[f"{s}_{j}_{n}"] for n in ("X", "Y", "Xv", "Yv"))
        mu, sd = float(X.mean()), float(X.std()) or 1.0
        f = lambda a: (a - mu) / sd
        keras.backend.clear_session()
        keras.utils.set_random_seed(t["seed"] + j)
        m = _build(t["arch"], t["mask"], L, H)
        hist, hit = _fit(m, f(X), f(Y), f(Xv), f(Yv), CAP_FOLD_S[s])
        back = lambda a: inv(a.astype("float64") * sd + mu)
        Ptr, Pva = back(m.predict(f(X), verbose=0)), back(m.predict(f(Xv), verbose=0))
        Ytr, Yva = inv(Y.astype("float64")), inv(Yv.astype("float64"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = MLPRegressor(random_state=t["seed"] + j, max_iter=200).fit(f(X), f(Y))
        Pmlp = back(sk.predict(f(Xv)).reshape(len(Xv), H))
        Pnaive = np.repeat(inv(Xv[:, -1:].astype("float64")), H, axis=1)   # último valor observado
        res_h1 = (Yva - Pva)[:, 0]
        folds.append(dict(fold=j, n_train=len(X), n_val=len(Xv), truncated=bool(hit),
                          keras_val=_metrics(Pva, Yva), keras_train=_metrics(Ptr, Ytr),
                          mlpreg_val=_metrics(Pmlp, Yva), naive_val=_metrics(Pnaive, Yva),
                          bds_p_h1=_bds_p(res_h1), train_curve=[float(v) for v in hist["rmse"]],
                          val_curve=[float(v) for v in hist["val_rmse"]]))
        arrays.update({f"f{j}_tr_pred": Ptr.astype("float32"), f"f{j}_tr_real": Ytr.astype("float32"),
                       f"f{j}_tr_start": _starts(z32, X), f"f{j}_va_pred": Pva.astype("float32"),
                       f"f{j}_va_real": Yva.astype("float32"), f"f{j}_va_start": _starts(z32, Xv),
                       f"f{j}_mlpreg_va_pred": Pmlp.astype("float32")})
    base = f"{OUT}/preds/detail_L{L}_H{H}_{s}"
    os.makedirs(f"{OUT}/preds", exist_ok=True)
    np.savez_compressed(base + ".tmp.npz", lag=L, **arrays)
    os.replace(base + ".tmp.npz", base + ".npz")
    return dict(kind="detail", strat=s, n_folds=nf, folds=folds, preds_file=base + ".npz",
                secs=time.time() - t0)


def run_task(t):
    if time.time() > t["deadline"]:
        return {"id": t["id"], "skipped": True}
    try:
        r = {"cv": _task_cv, "final": _task_final, "detail": _task_detail}[t["kind"]](t)
        r.update(id=t["id"], phase=t["phase"], L=t["L"], H=t["H"],
                 arch=[int(a) for a in t["arch"]], mask=[bool(m) for m in t["mask"]], finished=time.time())
        return r
    except Exception:
        return {"id": t["id"], "error": traceback.format_exc()}


# ---------------- proceso principal ----------------
def _seed(i):
    return int(hashlib.md5(i.encode()).hexdigest()[:8], 16) % (2 ** 31 - 100)


def cv_task(phase, L, H, s, arch, mask):
    arch = tuple(int(a) for a in arch); mask = tuple(bool(m) for m in mask)
    i = f"{phase}|L{L}|H{H}|{s}|{arch}|{mask}"
    return dict(id=i, kind="cv", phase=phase, L=L, H=H, strat=s, arch=arch, mask=mask, seed=_seed(i))


def detail_task(L, H, s, arch, mask):
    arch = tuple(int(a) for a in arch); mask = tuple(bool(m) for m in mask)
    i = f"detail|L{L}|H{H}|{s}|{arch}|{mask}"
    return dict(id=i, kind="detail", phase="detail", L=L, H=H, strat=s, arch=arch, mask=mask, seed=_seed(i))


def final_task(phase, L, H, arch, mask, save=None):
    arch = tuple(int(a) for a in arch); mask = tuple(bool(m) for m in mask)
    i = f"{phase}|L{L}|H{H}|all|{arch}|{mask}"
    return dict(id=i, kind="final", phase=phase, L=L, H=H, arch=arch, mask=mask, seed=_seed(i), save=save)


def load_records():
    recs = {}
    for f in sorted(glob.glob(f"{RAW}/*.jsonl")):
        with open(f) as h:
            for line in h:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line); recs[r["id"]] = r
                    except Exception:
                        pass
    return recs


class Writer:
    def __init__(self):
        self.buf, self.t, self.n = [], time.time(), 0
        self.tag = f"{int(time.time())}_{uuid.uuid4().hex[:10]}"   # único: nunca sobrescribe otro bloque

    def add(self, r):
        self.buf.append(r)
        if time.time() - self.t > FLUSH_S:
            self.flush()

    def flush(self):
        if self.buf:
            atomic_text(f"{RAW}/chunk_{self.tag}_{self.n:05d}.jsonl",
                        "\n".join(json.dumps(r) for r in self.buf) + "\n")
            self.n += 1; self.buf = []
        self.t = time.time()


def intercalar(listas):
    """Round-robin entre configuraciones: si el tope corta F4, cada configuración queda con una
    muestra aleatoria de tamaño similar, en vez de configuraciones completas y otras sin evaluar."""
    return [t for grupo in itertools.zip_longest(*listas) for t in grupo if t is not None]


def run_phase(name, tasks, deadline):
    recs = load_records()
    todo = [dict(t, deadline=deadline) for t in tasks if t["id"] not in recs]
    log(f"{name}: {len(tasks)} tareas, {len(tasks) - len(todo)} ya hechas, {len(todo)} pendientes")
    if not todo:
        return
    if time.time() > deadline:
        log(f"{name}: tope de tiempo ya alcanzado; no se inicia"); return
    W, ok, skip, err, t0 = Writer(), 0, 0, 0, time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, maxtasksperchild=100) as pool:
        for r in pool.imap_unordered(run_task, todo, chunksize=1):
            if r.get("skipped"):
                skip += 1; continue
            if "error" in r:
                err += 1
                atomic_text(f"{ERR}/{hashlib.md5(r['id'].encode()).hexdigest()}.txt", r["id"] + "\n" + r["error"])
                if err >= 20 and ok == 0:
                    W.flush(); log(f"{name}: 20 errores sin ningún éxito; se aborta. Ver {ERR}"); pool.terminate()
                    raise SystemExit(1)
                continue
            W.add(r); ok += 1
            if ok % 25 == 0 or ok == len(todo):
                rate = ok / (time.time() - t0)
                log(f"{name}: {ok}/{len(todo)} | {rate*60:.1f} tareas/min | "
                    f"faltan ~{(len(todo)-ok)/rate/3600:.2f} h al ritmo actual | errores {err}")
    W.flush()
    log(f"{name}: terminadas {ok}, saltadas por tope {skip}, errores {err}, {(time.time()-t0)/60:.1f} min")


def ranking(recs, L, H):
    c = [r for r in recs.values() if r.get("phase") == "search" and r["L"] == L and r["H"] == H
         and not r.get("truncated")]
    bal = [r for r in c if r.get("gap_ratio") is not None and r["gap_ratio"] < GAP_MAX]
    return sorted(bal if bal else c, key=lambda r: r["best_val_rmse"]), bool(bal), len(c)


def write_summary(recs):
    out = {}
    for L, H in ALL_CONFIGS:
        top, bal, n = ranking(recs, L, H)
        out[f"L{L}_H{H}"] = dict(evaluados=n, total=len(shuffled(L, H)), hay_balanceados=bal,
                                 ganadora=(dict(arch=top[0]["arch"], mask=top[0]["mask"],
                                                best_val_rmse=top[0]["best_val_rmse"],
                                                gap_ratio=top[0]["gap_ratio"]) if top else None))
    atomic_text(f"{OUT}/resumen_busqueda.json", json.dumps(out, indent=1))


def main():
    T0 = float(open(T0_FILE).read()) if os.path.exists(T0_FILE) else None
    if T0 is None:
        T0 = time.time(); atomic_text(T0_FILE, str(T0))
    kill_at = T0 + H_KILL * 3600
    if time.time() >= kill_at:
        log("Ya pasaron 18 h desde T0: no se entrena más."); return

    def _kill():
        log("TOPE DE 18 h: se matan el proceso y sus hijos."); sys.stdout.flush()
        if os.getpgid(0) == os.getpid():
            os.killpg(os.getpgid(0), signal.SIGKILL)
        os._exit(1)
    tm = threading.Timer(kill_at - time.time(), _kill); tm.daemon = True; tm.start()
    stop_new, stop_search = T0 + H_STOP_NEW * 3600, T0 + H_STOP_SEARCH * 3600
    log(f"INICIO pid={os.getpid()} smoke={SMOKE} | T0={time.ctime(T0)} | sin tareas nuevas desde "
        f"{time.ctime(stop_new)} | kill {time.ctime(kill_at)}")

    for L, H in ALL_CONFIGS:
        prep_splits(L, H)

    L, H = REP
    run_phase("F1 búsqueda (14,14)", [cv_task("search", L, H, "gk", a, m) for a, m in shuffled(L, H)], stop_new)
    recs = load_records(); write_summary(recs)
    top, bal, n = ranking(recs, *REP)
    if not top:
        log("F1 sin resultados válidos; se detiene."); return
    wa, wm = top[0]["arch"], top[0]["mask"]
    log(f"Ganadora F1: {wa} {wm} (balanceada={bal}, evaluadas={n})")

    run_phase("F2 ganadora F1 x 16 config x 3 estrategias",
              [cv_task("eval_rep", L, H, s, wa, wm) for L, H in ALL_CONFIGS for s in ("gk", "fc", "kf")], stop_new)
    run_phase("F3 modelos finales con ganadora F1",
              [final_task("final_rep", L, H, wa, wm, save="rep") for L, H in ALL_CONFIGS], stop_new)
    run_phase("F4 búsqueda en las otras configuraciones",
              intercalar([[cv_task("search", L, H, "gk", a, m) for a, m in shuffled(L, H)] for L, H in F4_ORDER]),
              stop_search)

    recs = load_records(); write_summary(recs)
    p1, p2, p3 = [], [], []
    for L, H in ALL_CONFIGS:
        top, bal, n = ranking(recs, L, H)
        if not top:
            continue
        p1.append(final_task("top5", L, H, top[0]["arch"], top[0]["mask"], save="own"))
        p2 += [cv_task("eval_own", L, H, s, top[0]["arch"], top[0]["mask"]) for s in ("fc", "kf")]
        p3 += [final_task("top5", L, H, r["arch"], r["mask"]) for r in top[1:5]]
    run_phase("F5 ganadoras propias (final, fc/kf, Top-5 hold-out)", p1 + p2 + p3, stop_new)
    recs = load_records(); write_summary(recs)
    p6 = []
    for L, H in ALL_CONFIGS:
        top, bal, n = ranking(recs, L, H)
        a, m = (top[0]["arch"], top[0]["mask"]) if top else (wa, wm)
        p6 += [detail_task(L, H, s, a, m) for s in ("gk", "fc", "kf")]
    run_phase("F6 detalle por fold (predicciones, métricas USD, BDS, baselines)", p6, stop_new)
    write_summary(load_records())
    log("FIN: todas las fases terminaron o alcanzaron su tope.")


if __name__ == "__main__":
    main()
