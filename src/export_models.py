"""Exporta a app/ el mejor modelo por horizonte (un model_h{H}.keras y un scalers_h{H}.joblib).

Para cada horizonte H elegimos el lag cuyo modelo ganador tiene el menor RMSE de validación
en USD con Group K-Fold (fase F6, promedio de 5 folds). La elección usa solo validación;
el hold-out no interviene. Si F6 no existe para algún H, se usa el modelo F3 (ganadora de F1).

Uso (Colab o local):  python src/export_models.py --results /ruta/a/results --app app
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

LAGS, HORIZONS = (7, 14, 21, 28), (7, 14, 21, 28)


def scaler(n, mu, sd):
    s = StandardScaler()
    s.mean_, s.scale_ = np.full(n, mu), np.full(n, sd)
    s.var_, s.n_features_in_, s.n_samples_seen_ = s.scale_ ** 2, n, 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--app", default="app")
    a = ap.parse_args()
    res, app_dir = Path(a.results), Path(a.app)
    recs = [json.loads(line) for f in glob.glob(str(res / "raw" / "*.jsonl")) for line in open(f) if line.strip()]
    detail = {(r["L"], r["H"]): r for r in recs if r.get("phase") == "detail" and r.get("strat") == "gk"}
    elegidos = {}
    for H in HORIZONS:
        cand = [(np.mean([f["keras_val"]["rmse"] for f in detail[(L, H)]["folds"]]), L)
                for L in LAGS if (L, H) in detail and (res / "models" / f"own_L{L}_H{H}.keras").exists()]
        if cand:
            rmse, L = min(cand)
            tag = "own"
        else:
            L, rmse, tag = 14, None, "rep"
        base = res / "models" / f"{tag}_L{L}_H{H}"
        meta = json.loads(Path(str(base) + "_scaler.json").read_text())
        shutil.copyfile(str(base) + ".keras", app_dir / f"model_h{H}.keras")
        joblib.dump(dict(x=scaler(L, meta["mu"], meta["sd"]), y=scaler(H, meta["mu"], meta["sd"]),
                         use_log=meta["use_log"], lag=L, arch=meta["arch"], mask=meta["mask"]),
                    app_dir / f"scalers_h{H}.joblib")
        elegidos[H] = dict(lag=L, origen=tag, val_rmse_usd_gk=rmse, arch=meta["arch"], mask=meta["mask"])
        print(f"H={H}: lag={L} origen={tag} arch={meta['arch']} mask={meta['mask']} val_rmse_gk={rmse}")
    (app_dir / "modelos_exportados.json").write_text(json.dumps(elegidos, indent=1))


if __name__ == "__main__":
    main()
