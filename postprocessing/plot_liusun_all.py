"""Driver: regenerate all analytical Liu & Sun (2020) ILS-MPM re-plots.

Renders the contact-mechanics figures (geometry schematics, stress fields, and
traction/force curves) from the verified closed-form solutions in
postprocessing/contact_fields.py. Analytical reference only -- numerical-overlay
slots are left in the line plots for the implementation's results.

Run:  python3 postprocessing/plot_liusun_all.py
Outputs: figures/liusun_fig{12,13,14,15,16,21,22,23}_*_pub.png
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_liusun_hertz as hertz          # noqa: E402
import plot_liusun_nine_disc as nine       # noqa: E402
import plot_liusun_brazilian as braz       # noqa: E402


def main():
    print("Hertz (Fig.12, 13, 14):")
    hertz.main()
    print("Nine-disc (Fig.15, 16):")
    nine.main()
    print("Brazilian (Fig.21, 22, 23):")
    braz.main()
    print("\nDone. Figures in figures/liusun_*_pub.png")


if __name__ == "__main__":
    main()
