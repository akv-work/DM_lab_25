"""
Лабораторная работа №3. Скорость сходимости по сохранённым кривым валидации
(запускать после lab3_protein.py и lab3_wdbc.py).
"""
import numpy as np

from common import RESULTS_DIR

CASES = (("protein", "Protein (MSE, станд. RMSD)", 0.55), ("wdbc", "WDBC (Cross-Entropy)", 0.20))
MODES = (("plain", "Без предобучения"), ("pre", "С предобучением"))


def main():
    lines = []
    for key, title, thr in CASES:
        z = np.load(RESULTS_DIR / f"{key}_val_curves.npz")
        lines.append(f"{title}: минимум val loss и число эпох до val loss <= {thr}")
        for mode, name in MODES:
            curves = [z[k] for k in z.files if k.startswith(mode)]
            mins = [c.min() for c in curves]
            reached = [int(np.argmax(c <= thr)) for c in curves if (c <= thr).any()]
            lines.append(f"  {name:18s}: min val loss = {np.mean(mins):.4f} ± {np.std(mins, ddof=1):.4f}; "
                         f"эпох до порога = {np.mean(reached):.1f} (достигли порога {len(reached)} из {len(curves)} запусков)")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "convergence.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
