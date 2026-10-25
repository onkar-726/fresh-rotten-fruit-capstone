"""Build the PDF user manual (and its figures).   python tools/build_manual.py

Everything that can drift from the code (CLI options) is read from the real scripts' --help output.
No accuracy or training numbers appear in the manual: they exist only after YOU run the project.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "Fruit_Classification_User_Manual.pdf"
FIG.mkdir(parents=True, exist_ok=True)

COL = {"colab": "#DCEBFA", "colab_e": "#2B6CB0", "dgx": "#DFF3E3", "dgx_e": "#2F855A",
       "lap": "#FDE9D4", "lap_e": "#C05621", "ink": "#1A202C", "grey": "#718096"}


# ------------------------------------------------------------------ figures
def box(ax, x, y, w, h, text, fc, ec, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=COL["ink"],
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color="#4A5568", style="-|>", lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 9.4); ax.axis("off")
    lanes = [("GOOGLE COLAB   (data preparation)", 6.35, "colab"), ("COLLEGE NVIDIA DGX B200   (training and experiments)", 3.35, "dgx"),
             ("YOUR LAPTOP / DGX   (final application)", 0.35, "lap")]
    for title, y, k in lanes:
        ax.add_patch(Rectangle((0.1, y), 15.3, 2.75, fc=COL[k], ec=COL[k + "_e"], lw=1.2, alpha=0.35))
        ax.text(0.3, y + 2.5, title, fontsize=10.5, fontweight="bold", color=COL[k + "_e"])
    X = lambda c: 0.35 + c * 2.52  # noqa: E731
    W, H = 2.2, 1.35
    colab = ["Kaggle dataset\n(6 classes)", "Explore +\nvalidate pre-processing\n(notebook 01)", "Augmentation\npreview", "70 / 15 / 15 split\n+ verification", "Leakage audit\n(random vs time-block)", "fruit_data_bundle.zip\n(downloaded to your PC)"]
    for c, t in enumerate(colab):
        box(ax, X(c), 6.55, W, H, t, "white", COL["colab_e"])
        if c:
            arrow(ax, (X(c - 1) + W, 7.22), (X(c), 7.22))
    dgx = ["unzip + verify_data\n+ check_gpu", "Model 1\nBaseline CNN\n(all GPUs)", "Model 2\nMobileNetV2\n2-stage fine-tuning", "Evaluate\nvalidation + test\n95% CI, McNemar", "Compare, targets\nPASS / FAIL,\nerror analysis", "dgx_results.zip\n(models + results)"]
    for j, t in enumerate(dgx):
        c = 5 - j
        box(ax, X(c), 3.55, W, H, t, "white", COL["dgx_e"])
        if j:
            arrow(ax, (X(c + 1), 4.22), (X(c) + W, 4.22))
    arrow(ax, (X(5) + W / 2, 6.55), (X(5) + W / 2, 4.9), color=COL["colab_e"], lw=2)
    ax.text(X(5) + W / 2 + 0.12, 5.7, "scp  (step B)", fontsize=8.5, color=COL["colab_e"], fontweight="bold")
    lap = ["unzip results\n+ pip install", "python app/app.py\n(Gradio server)", "User uploads\na fruit photo", "Predicted class +\nfresh / rotten +\nconfidence score"]
    for c, t in enumerate(lap):
        box(ax, X(c), 0.55, W, H, t, "white", COL["lap_e"], bold=(c == 3))
        if c:
            arrow(ax, (X(c - 1) + W, 1.22), (X(c), 1.22))
    arrow(ax, (X(0) + W / 2, 3.55), (X(0) + W / 2, 1.9), color=COL["dgx_e"], lw=2)
    ax.text(X(0) + W / 2 + 0.12, 2.75, "scp  (step F)", fontsize=8.5, color=COL["dgx_e"], fontweight="bold")
    ax.text(X(5), 1.15, "Everything is saved locally.\nNo Google Drive is used.", fontsize=9, color=COL["grey"], style="italic")
    ax.set_title("Figure 1 - System architecture and where each part runs", fontsize=12, fontweight="bold", loc="left")
    fig.savefig(FIG / "fig1_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def chain(ax, y, items, fs=8):
    x = 0.2
    for text, w, fc, ec in items:
        box(ax, x, y, w, 1.05, text, fc, ec, fs=fs)
        if x > 0.2:
            arrow(ax, (x - 0.18, y + 0.52), (x, y + 0.52), lw=1.1)
        x += w + 0.2
    return x


def fig_models():
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 20.5); ax.set_ylim(0, 8.7); ax.axis("off")
    g, b, o, p = ("#E2E8F0", "#718096"), ("#DCEBFA", "#2B6CB0"), ("#FEEBC8", "#C05621"), ("#E9D8FD", "#6B46C1")
    ax.text(0.2, 8.15, "Model 1 - Baseline CNN (from scratch, ~1.2 M parameters)", fontsize=11, fontweight="bold")
    chain(ax, 6.65, [("Input\n224x224x3\n(float 0-255)", 1.9, *g), ("Rescale\n1/255", 1.3, *g), ("Block 1\n2x Conv-BN-ReLU\n32 + MaxPool", 2.2, *b),
                     ("Block 2\n64 filters", 1.7, *b), ("Block 3\n128 filters", 1.7, *b), ("Block 4\n256 filters", 1.7, *b),
                     ("Global\nAvg Pool", 1.4, *o), ("Dropout\n0.3", 1.2, *o), ("Dense 6\nsoftmax", 1.4, *p)])
    ax.text(0.2, 5.75, "All layers trainable; Adam lr 1e-3; class weights; augmentation; early stopping.", fontsize=8.5, color=COL["grey"])
    ax.text(0.2, 4.95, "Model 2 - MobileNetV2 transfer learning (ImageNet weights)", fontsize=11, fontweight="bold")
    chain(ax, 3.45, [("Input\n224x224x3\n(float 0-255)", 1.9, *g), ("Rescale\n[-1, 1]", 1.3, *g), ("MobileNetV2 backbone\n(inverted residual blocks)\nSTAGE 1: frozen", 3.3, "#C6F6D5", "#2F855A"),
                     ("Global\nAvg Pool", 1.4, *o), ("Dropout\n0.3", 1.2, *o), ("Dense 6\nsoftmax", 1.4, *p)])
    ax.text(0.2, 2.55, "Stage 1: backbone frozen, only the new head trains (lr 1e-3).", fontsize=8.5, color=COL["grey"])
    ax.text(0.2, 1.95, "Stage 2: top 40 backbone layers unfrozen (lr 2e-5); BatchNorm layers stay frozen and in inference mode.", fontsize=8.5, color=COL["grey"])
    ax.add_patch(Rectangle((0.2, 0.25), 0.4, 0.3, fc="#C6F6D5", ec="#2F855A")); ax.text(0.7, 0.33, "pretrained (frozen or fine-tuned)", fontsize=8.5)
    ax.add_patch(Rectangle((5.2, 0.25), 0.4, 0.3, fc=b[0], ec=b[1])); ax.text(5.7, 0.33, "trained from scratch", fontsize=8.5)
    ax.add_patch(Rectangle((8.9, 0.25), 0.4, 0.3, fc=o[0], ec=o[1])); ax.text(9.4, 0.33, "pooling / regularisation", fontsize=8.5)
    ax.add_patch(Rectangle((13.2, 0.25), 0.4, 0.3, fc=p[0], ec=p[1])); ax.text(13.7, 0.33, "6-class output", fontsize=8.5)
    ax.set_title("Figure 2 - The two model architectures", fontsize=12, fontweight="bold", loc="left")
    fig.savefig(FIG / "fig2_models.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_split():
    import numpy as np
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(11, 3.3))
    colors = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
    for row, cls in enumerate(["class A", "class B", "class C"]):
        y = 2 - row
        assign = ["train", "train", "validation", "train", "test", "train", "train", "test", "train", "validation"]
        rng.shuffle(assign)
        for i, s in enumerate(assign):
            ax.add_patch(Rectangle((i * 1.0, y), 0.86, 0.7, fc=colors[s], ec="white"))
            if i and assign[i - 1] != s:
                ax.add_patch(Rectangle((i * 1.0 - 0.14, y), 0.14, 0.7, fc="white", ec="#A0AEC0", hatch="///", lw=0.5))
        ax.text(-0.15, y + 0.35, cls, ha="right", va="center", fontsize=9)
    ax.set_xlim(-1.3, 10.2); ax.set_ylim(-0.2, 2.95); ax.axis("off")
    ax.annotate("time ->  (capture time of the source photos)", (4.5, -0.15), ha="center", fontsize=9, color=COL["grey"])
    for k, (s, c) in enumerate(colors.items()):
        ax.add_patch(Rectangle((1.3 + k * 2.2, 2.85), 0.3, 0.2, fc=c)); ax.text(1.7 + k * 2.2, 2.9, s, fontsize=9, va="center")
    ax.add_patch(Rectangle((8.0, 2.85), 0.3, 0.2, fc="white", ec="#A0AEC0", hatch="///")); ax.text(8.4, 2.9, "time buffer", fontsize=9, va="center")
    ax.set_title("Figure 3 - Time-block split (schematic illustration, not real data)", fontsize=11, fontweight="bold", loc="left")
    fig.savefig(FIG / "fig3_split.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_app():
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.add_patch(Rectangle((0.1, 0.1), 9.8, 4.8, fc="white", ec="#4A5568", lw=1.5))
    ax.text(0.4, 4.45, "Fresh or Rotten?", fontsize=15, fontweight="bold")
    ax.text(0.4, 4.1, "Upload a photo of an apple, banana or orange.", fontsize=9, color=COL["grey"])
    ax.add_patch(Rectangle((0.4, 1.4), 4.2, 2.45, fc="#F7FAFC", ec="#A0AEC0", ls="--"))
    ax.text(2.5, 2.65, "drop a fruit photo here\nor click to upload", ha="center", fontsize=10, color=COL["grey"])
    ax.add_patch(Rectangle((0.4, 0.95), 0.22, 0.22, fc="white", ec="#4A5568")); ax.text(0.75, 0.97, "show where the model looked (Grad-CAM)", fontsize=8.5)
    box(ax, 0.4, 0.25, 1.9, 0.5, "Classify", "#FDBA74", "#C05621", fs=10, bold=True)
    ax.text(5.0, 3.7, "<Fruit> - FRESH / ROTTEN", fontsize=13, fontweight="bold", color="#2F855A")
    ax.text(5.0, 3.25, "Predicted class: <one of the six classes>\nConfidence: xx.x %\nFresh/rotten confidence: xx.x %", fontsize=9.5, va="top", linespacing=1.5)
    ax.text(5.0, 2.2, "Probability of each class", fontsize=9, fontweight="bold")
    for i in range(4):
        ax.add_patch(Rectangle((5.0, 1.75 - i * 0.38), 3.6 - i * 0.8, 0.22, fc="#4C72B0" if i == 0 else "#CBD5E0"))
    ax.text(8.9, 0.5, "(Grad-CAM image)", fontsize=8, color=COL["grey"], ha="center")
    ax.set_title("Figure 4 - Layout of the application (schematic; the values shown are placeholders)", fontsize=11, fontweight="bold", loc="left")
    fig.savefig(FIG / "fig4_app.png", dpi=170, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ PDF
def build_pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                    Preformatted, Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                    ("DVM", "DejaVuSansMono.ttf"), ("DVM-B", "DejaVuSansMono-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fd / f)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
    pdfmetrics.registerFontFamily("DVM", normal="DVM", bold="DVM-B", italic="DVM", boldItalic="DVM-B")

    NAVY, TEAL = colors.HexColor("#1A365D"), colors.HexColor("#2C7A7B")
    S = {
        "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=NAVY, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=colors.HexColor("#4A5568"), alignment=TA_CENTER),
        "H1": ParagraphStyle("H1", keepWithNext=1, fontName="DV-B", fontSize=17, leading=22, textColor=NAVY, spaceBefore=10, spaceAfter=8),
        "H2": ParagraphStyle("H2", keepWithNext=1, fontName="DV-B", fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=10, spaceAfter=4),
        "H3": ParagraphStyle("H3", keepWithNext=1, fontName="DV-B", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle("body", fontName="DV", fontSize=9.3, leading=13.2, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.1, leading=10.6),
        "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.1, leading=10.6, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=9.6, backColor=colors.HexColor("#F1F5F9"),
                               borderColor=colors.HexColor("#CBD5E0"), borderWidth=0.5, borderPadding=5, spaceBefore=3, spaceAfter=8, leftIndent=4),
        "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.3, leading=13, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8, leading=10, textColor=colors.HexColor("#4A5568"), alignment=TA_CENTER, spaceAfter=8),
    }
    W = A4[0] - 3.4 * cm

    def P(t, s="body"): return Paragraph(t, S[s])
    def H1(t): return Paragraph(t, S["H1"])
    def H2(t): return Paragraph(t, S["H2"])
    def bullets(items): return [Paragraph(i, S["bullet"], bulletText="•") for i in items]
    _CMD = {"python", "python3", "scp", "ssh", "docker", "pip", "bash", "unzip", "export", "sbatch", "tmux", "watch",
            "cd", "nvidia-smi", "tail", "make", "source", "set"}
    CODE_LIMIT = 96          # characters that fit the code box without clipping (7.6 pt mono on A4)

    def wrap_code(text, limit=CODE_LIMIT):
        """Wrap over-long lines. Shell commands get a trailing backslash, so copy-paste still works."""
        import re as _re
        out = []
        for line in text.split("\n"):
            if len(line) <= limit:
                out.append(line)
                continue
            body, comment = line, ""
            m = _re.search(r"\s{2,}#", line)
            if m and m.start() > 0:
                body, comment = line[:m.start()], line[m.start():].strip()
            first = body.split(" ")[0] if body.strip() else ""
            shell = first in _CMD
            cur = ""
            for word in body.split(" "):
                if cur and len(cur) + 1 + len(word) > limit - 2:
                    out.append(cur + (" \\" if shell else ""))
                    cur = "    " + word
                else:
                    cur = (cur + " " + word) if cur else word
            out.append(cur)
            if comment:
                if len(out[-1]) + 2 + len(comment) <= limit:
                    out[-1] += "  " + comment
                else:
                    out.append(comment)
        return "\n".join(out)

    def code(t):
        text = wrap_code(t.strip("\n"))
        too_long = [l for l in text.split("\n") if len(l) > CODE_LIMIT + 4]
        assert not too_long, f"code line would be clipped in the PDF: {too_long[0]}"
        return Preformatted(text, S["code"])

    def callout(kind, text):
        palette = {"RUN ON": ("#EBF8FF", "#2B6CB0"), "EXPECTED OUTPUT": ("#F0FFF4", "#2F855A"), "WARNING": ("#FFF5F5", "#C53030"),
                   "TIP": ("#FFFAF0", "#C05621"), "NOT TESTED HERE": ("#FAF5FF", "#6B46C1")}
        bg, ec = palette[kind]
        t = Table([[Paragraph(f"<b>{kind}</b><br/>{text}", S["cell"])]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(ec)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return [t, Spacer(1, 6)]

    def table(rows, widths, header=True, zebra=True):
        data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        if zebra:
            st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")) for i in range(2, len(rows), 2)]
        t.setStyle(TableStyle(st))
        return [t, Spacer(1, 8)]

    def figure(name, caption, width=1.0):
        from PIL import Image as PILImage
        w, h = PILImage.open(FIG / name).size
        iw = W * width
        return [Image(str(FIG / name), width=iw, height=iw * h / w), Paragraph(caption, S["caption"])]

    # ------------------------------------------------------------------ document class with TOC
    class Doc(BaseDocTemplate):
        def afterFlowable(self, fl):
            if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2"):
                level = 0 if fl.style.name == "H1" else 1
                text = fl.getPlainText()
                key = f"h{self.seq.nextf('toc')}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page, key))

    def cover_page(c, d):
        c.saveState()
        c.setFillColor(NAVY); c.rect(0, A4[1] - 3.2 * cm, A4[0], 3.2 * cm, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("DV-B", 11)
        c.drawString(1.7 * cm, A4[1] - 1.6 * cm, "RAMDEOBABA UNIVERSITY, NAGPUR  |  Department of Electronics Engineering")
        c.setFont("DV", 9.5); c.drawString(1.7 * cm, A4[1] - 2.3 * cm, "Deep Learning Lab  |  Capstone Project  |  Group 19  |  Session 2026-27, Semester V")
        c.restoreState()

    def normal_page(c, d):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#CBD5E0")); c.line(1.7 * cm, A4[1] - 1.3 * cm, A4[0] - 1.7 * cm, A4[1] - 1.3 * cm)
        c.setFont("DV", 8); c.setFillColor(colors.HexColor("#4A5568"))
        c.drawString(1.7 * cm, A4[1] - 1.1 * cm, "Fresh-versus-Rotten Fruit Classification - User Manual")
        c.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, f"Page {d.page}")
        c.drawString(1.7 * cm, 1.1 * cm, "Group 19 capstone")
        c.restoreState()

    doc = Doc(str(OUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.9 * cm, bottomMargin=1.7 * cm,
              title="Fresh-versus-Rotten Fruit Classification - User Manual", author="Group 19", subject="Capstone project instruction manual")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate("cover", [fr], onPage=cover_page), PageTemplate("normal", [fr], onPage=normal_page)])

    from reportlab.platypus import NextPageTemplate
    story = []

    # ---------------- cover
    story += [Spacer(1, 4.5 * cm), P("Fresh-versus-Rotten<br/>Fruit Classification", "title"), Spacer(1, 0.5 * cm),
              P("Complete Instruction Manual", "subtitle"), Spacer(1, 0.3 * cm),
              P("Baseline CNN  vs  MobileNetV2 transfer learning<br/>Colab data preparation  -  DGX B200 training  -  Gradio application", "subtitle"),
              Spacer(1, 1.2 * cm)]
    story += table([["Document", "What it covers"],
                    ["This manual", "which file to open, where to run it (Colab / DGX B200 / laptop), exact commands, expected outputs, troubleshooting, final checklist"],
                    ["README.md", "short overview and quick start"],
                    ["docs/", "METHODOLOGY, DGX_B200_GUIDE, DESIGN_DECISIONS, PROPOSAL_TRACEABILITY, TROUBLESHOOTING, TESTING_STATUS"],
                    ["notebooks/", "four executable notebooks (01 Colab, 02-04 DGX or laptop)"]], [0.22, 0.78])
    story += callout("WARNING", "This project package contains <b>no trained models and no accuracy, precision, recall or F1 numbers</b>. "
                     "They appear only after you run the training on the DGX. The code checks the proposal targets (accuracy at least 90 %, "
                     "macro-F1 above 0.85) against your real results and prints PASS or FAIL - it never assumes success.")
    story += [NextPageTemplate("normal"), PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t1", fontName="DV-B", fontSize=9.5, leading=12.5, leftIndent=0, spaceBefore=1.5),
                       ParagraphStyle("t2", fontName="DV", fontSize=8.5, leading=10.3, leftIndent=14)]
    story += [H1("Contents"), toc, PageBreak()]

    # ---------------- 1 overview
    story += [H1("1. Project at a glance")]
    story += [P("The system takes a photo of a fruit and answers three things: <b>which of six classes</b> it belongs to (fresh/rotten apple, banana, orange), "
                "<b>whether the fruit is fresh or rotten</b>, and <b>how confident</b> the model is. Two models are trained on the same data and compared fairly: "
                "Model 1 is a CNN built from scratch (the baseline); Model 2 is MobileNetV2 with transfer learning. The proposal targets are "
                "at least 90 % accuracy and an F1-score above 0.85 for both models - these are tested against the real results.")]
    story += table([["Phase", "Where", "What you do", "Main file"],
                    ["A. Data preparation", "<b>Google Colab</b>", "download, explore, validate, preview augmentation, split 70/15/15, verify, make bundle", "notebooks/01_colab_data_preparation.ipynb"],
                    ["B. Transfer", "your computer", "move the bundle to the DGX and check it arrived intact", "scp, scripts/verify_data.py"],
                    ["C. Training", "<b>DGX B200</b>", "train both models on all suitable GPUs; optional experiments", "scripts/train.py or notebooks/02_dgx_training.ipynb"],
                    ["D. Evaluation", "DGX", "metrics, confusion matrices, fair comparison, PASS/FAIL, error analysis", "scripts/evaluate.py, compare.py, error_analysis.py, notebooks 03-04"],
                    ["E. Application", "laptop (or DGX)", "upload an image, see class, fresh/rotten status and confidence", "app/app.py"]],
                   [0.17, 0.16, 0.40, 0.27])
    story += figure("fig1_architecture.png", "Figure 1 - Architecture of the whole system and where every part runs.")
    story += [P("<b>Important assumptions.</b> The GPU specifications and the Colab/DGX workflow you mentioned were not available when this package was created. "
                "The package therefore assumes: a DGX B200 node with several NVIDIA B200 (Blackwell) GPUs, shell/SSH access, and an environment that you set up "
                "(NGC container or virtual environment). Nothing is hard-coded: GPU count, batch size and precision are command-line options "
                "(section 7 and docs/DGX_B200_GUIDE.md). If your site differs (for example Slurm, no internet), use the matching alternative given in each step.")]

    # ---------------- 2 architecture
    story += [PageBreak(), H1("2. Design in brief")]
    story += figure("fig2_models.png", "Figure 2 - Model 1 (baseline CNN) and Model 2 (MobileNetV2, two training stages).")
    story += [H2("2.1 Why the data split is special"),
              P("The Kaggle data holds only about 1,500 <i>original</i> photos; most of the roughly 13,600 files are rotated, flipped, shifted or noisy <i>copies</i>, "
                "and the originals are screenshots taken seconds apart. If files are split at random, copies and near-duplicates of the same fruit land in both "
                "training and test data and the accuracy looks excellent for the wrong reason. Notebook 01 re-measures these facts on your copy of the data. "
                "The project therefore splits <b>source photos in time blocks</b> with a 60-second gap between train, validation and test (seed 42). "
                "The result is close to - not exactly - 70/15/15 of the images, because whole blocks are assigned and the buffer removes some photos.")]
    story += figure("fig3_split.png", "Figure 3 - Time-block split: whole blocks go to one split; a buffer separates neighbours.", 0.9)
    story += [H2("2.2 How the models are judged"),
              *bullets(["<b>Primary test view:</b> one original image per source photo (independent samples). <b>Secondary:</b> all test images including augmented copies.",
                        "Accuracy, precision, recall, F1 (per class and macro), confusion matrices, 95 % bootstrap intervals, exact McNemar test.",
                        "Fairness: same split, seed, input size, augmentation, class weights, loss and optimiser for both models (a table proves it).",
                        "Size and speed: parameters, file size, latency per image, optional TensorFlow Lite export - the proposal asks which model suits limited hardware.",
                        "Final model for the application: chosen by a rule fixed in advance using the <b>validation</b> split only."])]

    # ---------------- 3 folder map
    story += [H1("3. What is in the project folder")]
    story += [code("""
fresh-rotten-fruit-capstone/
  README.md  Makefile  requirements*.txt  pyproject.toml  LICENSE
  configs/     default.json  dgx_b200.json  single_gpu_debug.json      <- hardware profiles
  notebooks/   01_colab_data_preparation.ipynb   (Colab)
               02_dgx_training.ipynb             (DGX)
               03_evaluation_comparison.ipynb    (DGX / laptop)
               04_error_analysis_and_app.ipynb   (DGX / laptop)
  scripts/     prepare_data  download_dataset  verify_data  check_gpu  fetch_pretrained_weights
               train  run_experiments  summarize_experiments  evaluate  compare  error_analysis
               export_tflite  predict  package_results  run_dgx_pipeline.sh
  slurm/       train_dgx.sbatch                  <- optional, only if your DGX uses Slurm
  app/         app.py                            <- the web application
  src/fruitfresh/   the Python package used by all scripts and notebooks
  tests/       automated tests (synthetic data)
  docs/        guides + this manual + figures
  data/splits/ split CSVs + split_manifest.json   (created in step A)
  data/raw/    dataset images on the DGX            (created in step B)
  workspace/   trained runs: models, histories, predictions   (created in step C)
  results/     tables, figures, reports, final_model.json     (created in step D)""")]

    # ---------------- 4 execution order
    story += [PageBreak(), H1("4. Complete execution order")]
    story += [P("Follow the steps in this order. Each row says <b>where</b> to run it and <b>which file</b> to open or which command to type. "
                "Details and expected outputs follow in sections 5-9.")]
    order = [["#", "Where", "File / command", "Result you should see"],
             ["A1", "Colab", "Upload <i>fresh-rotten-fruit-capstone.zip</i> to /content (Files panel); open <i>notebooks/01_colab_data_preparation.ipynb</i>", "notebook opens"],
             ["A2", "Colab", "Runtime > Run all (CPU is enough)", "last cell prints <i>READY FOR THE DGX</i>"],
             ["A3", "Colab", "browser downloads <i>fruit_data_bundle.zip</i>", "file on your computer"],
             ["B1", "your PC", "<font name='DVM'>scp fruit_data_bundle.zip user@dgx:path/fresh-rotten-fruit-capstone/</font>", "file copied"],
             ["B2", "DGX", "<font name='DVM'>unzip -o fruit_data_bundle.zip</font>", "data/splits/ (and data/raw/) exist"],
             ["B3", "DGX", "<font name='DVM'>python scripts/verify_data.py --split-dir data/splits --data-root data/raw</font>", "<i>DATA OK</i>"],
             ["C1", "DGX", "create the environment (section 7.1), then <font name='DVM'>python scripts/check_gpu.py</font>", "<i>GPU CHECK PASSED</i>"],
             ["C2", "DGX (offline only)", "<font name='DVM'>python scripts/fetch_pretrained_weights.py</font> on a PC with internet; copy assets/*.h5 to the DGX", "weights file in assets/"],
             ["D1", "DGX", "<font name='DVM'>python scripts/train.py --model baseline_cnn --config configs/default.json</font>", "<i>[baseline_cnn] done in ...</i>"],
             ["D2", "DGX", "<font name='DVM'>python scripts/train.py --model mobilenet_v2 --config configs/default.json</font>", "<i>[mobilenet_v2] done in ...</i>"],
             ["D3", "DGX (optional)", "<font name='DVM'>python scripts/run_experiments.py --family mobilenet_v2 --names head_only ft_top20</font>", "extra runs in workspace/runs/"],
             ["E1", "DGX", "<font name='DVM'>python scripts/evaluate.py</font>", "one line per model and test view"],
             ["E2", "DGX", "<font name='DVM'>python scripts/compare.py</font>", "metrics table, PASS/FAIL table, final model"],
             ["E3", "DGX", "<font name='DVM'>python scripts/error_analysis.py --run baseline_cnn</font> (and mobilenet_v2)", "results/error_analysis/..."],
             ["E4", "DGX", "<font name='DVM'>python scripts/package_results.py --out dgx_results.zip</font>", "dgx_results.zip"],
             ["F1", "your PC", "<font name='DVM'>scp user@dgx:path/.../dgx_results.zip .</font>; <font name='DVM'>unzip -o dgx_results.zip -d fresh-rotten-fruit-capstone</font>", "workspace/runs and results/ appear"],
             ["F2", "laptop", "<font name='DVM'>pip install -r requirements.txt</font>; <font name='DVM'>python app/app.py</font>", "browser: http://127.0.0.1:7860"],
             ["G", "anywhere", "go through the final checklist (section 11)", "all boxes ticked"]]
    story += table(order, [0.06, 0.15, 0.55, 0.24])
    story += callout("TIP", "Everything on the DGX can be run in one go with <font name='DVM'>bash scripts/run_dgx_pipeline.sh</font> (steps 3-8 above; optional environment variables PROFILE, DATA_ROOT, EXPERIMENTS=1, OUT_ZIP, ALLOW_CPU=1, EXTRA_ARGS). "
                     "Run long jobs inside <font name='DVM'>tmux</font> so a dropped SSH connection does not stop them. Finished runs are detected and never retrained by accident.")

    # ---------------- 5 Colab
    story += [PageBreak(), H1("5. Phase A - Google Colab: data preparation")]
    story += callout("RUN ON", "<b>Google Colab</b> (CPU runtime is enough). File: <b>notebooks/01_colab_data_preparation.ipynb</b>.")
    story += [H2("5.1 Open the notebook"),
              *bullets(["Go to colab.research.google.com. In the left <b>Files</b> panel upload <i>fresh-rotten-fruit-capstone.zip</i> (it must end up as /content/fresh-rotten-fruit-capstone.zip).",
                        "Upload the notebook too (File > Upload notebook) <b>or</b> open it from GitHub if you pushed the project there and set <font name='DVM'>REPO_URL</font> in the first cell.",
                        "The first cell (SETUP) unzips the project, installs the few extra packages (Colab already has TensorFlow) and prints the project paths."]),
              H2("5.2 Run all cells - what each section does"),
              ]
    story += table([["Section", "What happens", "What to look for"],
                    ["1 Download", "kagglehub downloads the public dataset", "prints <i>Dataset root: ...</i> and the folders train, test"],
                    ["2 Exploration", "index of every file: class, source photo, original or augmented copy, capture time", "files vs source photos: far fewer sources than files; class balance chart; sample images"],
                    ["3 Pre-processing", "unreadable-file check, image-size statistics, one tensor batch", "<i>Unreadable images: 0</i> (or a short list); batch shape (8, 224, 224, 3), range 0-255"],
                    ["4 Augmentation", "preview of the training-time augmentation", "grid of varied versions of one photo"],
                    ["5 Split", "70/15/15 time-block split, verification (raises an error if anything leaks)", "image and source counts per split, minimum gap at least 60 s, timeline plot"],
                    ["6 Leakage audit", "random split vs time-block split similarity", "two summary lines + histogram"],
                    ["7 Bundle", "saves data/splits/*.csv + manifest, creates fruit_data_bundle.zip, downloads it", "browser download; <i>READY FOR THE DGX</i>"]],
                   [0.17, 0.43, 0.40])
    story += callout("EXPECTED OUTPUT", "The achieved percentages of the split are printed by section 5 (images per split divided by the total). They should be close to 70 / 15 / 15; "
                     "a deviation of a few points is normal (whole blocks + time buffer). The numbers in your run are the ones that count.")
    story += callout("WARNING", "<b>INCLUDE_IMAGES</b> (last section): keep <b>True</b> if the DGX has no internet (the zip then contains the images - it can be large). "
                     "Set <b>False</b> to get a tiny zip with only the CSVs; the DGX then downloads the images itself (step 5.3).")
    story += [H2("5.3 Alternative: run without the notebook"),
              code("""
python scripts/prepare_data.py --data-root <folder with train/ and test/> --out data/splits --audit \\
       --bundle fruit_data_bundle.zip            # add --bundle-no-images for a CSV-only bundle""")]
    story += callout("NOT TESTED HERE", "Real Google Colab and the real Kaggle download could not be tested in the development environment. The same code was run on a synthetic dataset with identical structure and file naming; "
                     "see section 12.")

    # ---------------- 6 transfer
    story += [H1("6. Phase B - move the data to the DGX")]
    story += table([["Situation", "What to do"],
                    ["DGX has <b>no internet</b>", "Use the bundle <b>with images</b>: <font name='DVM'>scp fruit_data_bundle.zip user@dgx:path/fresh-rotten-fruit-capstone/</font>, then on the DGX <font name='DVM'>unzip -o fruit_data_bundle.zip</font>. Images appear in data/raw/, CSVs in data/splits/."],
                    ["DGX <b>has internet</b>", "Copy only the CSV bundle (INCLUDE_IMAGES = False), unzip it, then <font name='DVM'>python scripts/download_dataset.py --link data/raw</font>."],
                    ["Dataset already on the DGX", "<font name='DVM'>export FRUIT_DATA_ROOT=/path/containing/train-and-test</font> or pass <font name='DVM'>--data-root</font> to every script."]],
                   [0.25, 0.75])
    story += [P("<b>Always verify</b> after the transfer. It checks the SHA-256 hash of every split CSV against the manifest and that every listed image exists:"),
              code("python scripts/verify_data.py --split-dir data/splits --data-root data/raw          # add --decode to open every image (slow)")]
    story += callout("EXPECTED OUTPUT", "A small JSON summary (images and sources per split, <i>missing: 0</i>) followed by <b>DATA OK</b>. "
                     "Anything else prints <i>DATA PROBLEM: ...</i> and exits with an error - fix it before training (section 10).")

    # ---------------- 7 DGX
    story += [PageBreak(), H1("7. Phase C - DGX B200: environment and training")]
    story += callout("RUN ON", "<b>The college NVIDIA DGX B200</b>, from the project folder, in a terminal (SSH) or JupyterLab. Notebook alternative: <b>notebooks/02_dgx_training.ipynb</b>.")
    story += [H2("7.1 Environment (choose ONE route)")]
    story += table([["Route", "Commands", "When"],
                    ["A. NGC TensorFlow container", "<font name='DVM'>docker run --gpus all -it --rm --shm-size=16g -v \"$PWD\":/workspace -w /workspace &lt;ngc-tensorflow-image&gt; bash</font><br/><font name='DVM'>pip install -r requirements-train.txt</font>", "recommended for Blackwell GPUs; choose a release whose notes mention B200/Blackwell; your site may use Singularity/Enroot instead of Docker"],
                    ["B. virtual environment", "<font name='DVM'>python3 -m venv .venv &amp;&amp; source .venv/bin/activate</font><br/><font name='DVM'>pip install tensorflow &amp;&amp; pip install -r requirements-train.txt</font>", "only if that TensorFlow build supports the B200 (CUDA 12.8 or newer)"],
                    ["C. conda", "same idea as B inside a conda environment", "if your site prefers conda"]],
                   [0.2, 0.5, 0.3])
    story += callout("WARNING", "The authors could <b>not</b> test on a B200. If TensorFlow does not see the GPUs, or reports <i>no kernel image</i>, the TensorFlow build is not compatible with Blackwell - use route A "
                     "or ask the DGX administrators which TensorFlow/CUDA combination is supported.")
    story += [H2("7.2 Check the GPUs"), code("python scripts/check_gpu.py"),
              *callout("EXPECTED OUTPUT", "TensorFlow and CUDA versions, <i>GPUs visible to TensorFlow: N</i>, one line per GPU, <i>GPU:i matmul OK</i> for each, <i>MirroredStrategy all-reduce OK</i> when N is above 1, "
                       "a bfloat16 line, and finally <b>GPU CHECK PASSED</b>. Do not continue until you see it.")]
    story += [H2("7.3 Train Model 1 - baseline CNN"),
              code("python scripts/train.py --model baseline_cnn --config configs/default.json"),
              P("The first line printed is the hardware summary (devices, number of replicas, global batch size, precision). Then one line per epoch. At the end:"),
              code("[baseline_cnn] done in <minutes> min on <N> x GPU, best val_loss <value>"),
              P("Files created in <font name='DVM'>workspace/runs/baseline_cnn/seed42/</font>: <b>model.keras</b> (final model), <b>history.csv</b> (curves), <b>train_info.json</b> (settings, hardware, timings, library versions). "
                "Running the same command again prints <i>already trained</i> and does nothing (use <font name='DVM'>--force</font> to retrain).")]
    story += [H2("7.4 Train Model 2 - MobileNetV2"),
              code("python scripts/train.py --model mobilenet_v2 --config configs/default.json\n# offline DGX:  add  --weights-file assets/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"),
              P("Stage 1 (head only) runs first, then stage 2 (fine-tuning). <b>train_info.json</b> records, for each stage, how many layers and parameters were trainable and that "
                "<i>backbone_batchnorm_trainable</i> is 0. After stage 1 the model is saved as <i>stage1.keras</i>; if the job dies later, running the command again resumes from there.")]
    story += [H2("7.5 Hardware options"),
              *table([["Option", "Meaning", "Example"],
                      ["--gpus", "all (default), a number, or ids", "--gpus 4   or   --gpus 0,1"],
                      ["--per-gpu-batch-size", "images per GPU; global batch = this x number of GPUs", "--per-gpu-batch-size 16"],
                      ["--mixed-precision", "off (default), bf16, fp16", "--mixed-precision bf16"],
                      ["--lr-scaling", "none (default), sqrt, linear: scale learning rate with GPU count", "--lr-scaling sqrt"],
                      ["--config", "JSON profile; command-line options override it", "configs/dgx_b200.json"],
                      ["--set key=value", "override a model setting for one run (with --run-name)", "--set ft_layers=20 --run-name my_variant"]],
                     [0.25, 0.45, 0.30]),
              P("<b>Profiles:</b> <i>configs/default.json</i> (all GPUs, 32 images/GPU, full precision) - use it first. <i>configs/dgx_b200.json</i> (64 images/GPU, bfloat16) - a performance profile to try after the default works. "
                "<i>configs/single_gpu_debug.json</i> - one GPU, to tell model problems from multi-GPU problems. The dataset is small: one B200 is plenty; using all GPUs shortens epochs but gives few optimisation steps per epoch "
                "(see docs/DGX_B200_GUIDE.md).")]
    story += [H2("7.6 Slurm and long jobs"),
              code("tmux new -s fruit            # then run the commands inside; detach with Ctrl-b d\n# Slurm sites: edit the #SBATCH lines, then\nsbatch slurm/train_dgx.sbatch\ntail -f logs/fruit-train-<jobid>.out"),
              P("Watch the GPUs with <font name='DVM'>watch -n 2 nvidia-smi</font>: memory should be in use on every selected GPU.")]
    story += [H2("7.7 Optional experiments on the DGX"),
              code("python scripts/run_experiments.py --family mobilenet_v2 --names head_only ft_top20 ft_top80 ft_lr_1e-5 --config configs/default.json\npython scripts/summarize_experiments.py --family mobilenet_v2"),
              P("Each experiment changes <b>one</b> thing and is judged on the <b>validation</b> split only; the test split is never used to choose. "
                "Names are listed in <i>src/fruitfresh/config.py</i> (EXPERIMENTS). To compare the best run of each model run both summaries, then <font name='DVM'>python scripts/compare.py --use-best</font>.")]
    story += callout("NOT TESTED HERE", "Multi-GPU training was tested only with <b>two logical CPU replicas</b> (same MirroredStrategy code path, no NCCL). Real B200 execution, bfloat16 on B200, and timing were not tested.")

    # ---------------- 8 evaluation
    story += [PageBreak(), H1("8. Phase D - evaluation, comparison and error analysis")]
    story += callout("RUN ON", "<b>DGX</b> (or any machine that holds the trained runs). Notebooks: <b>03_evaluation_comparison.ipynb</b>, <b>04_error_analysis_and_app.ipynb</b>.")
    story += [code("python scripts/evaluate.py\npython scripts/compare.py\npython scripts/error_analysis.py --run baseline_cnn\npython scripts/error_analysis.py --run mobilenet_v2"),
              P("<b>evaluate.py</b> prints one line per model and test view: accuracy, macro-F1, number of images and source photos, and <b>targets: PASS/FAIL</b>. "
                "<b>compare.py</b> prints the metric table, the PASS/FAIL table against the proposal targets, and the final model for the application.")]
    story += table([["File (in results/ or the run's eval/ folder)", "Content"],
                    ["comparison_report.md", "all tables of the comparison, assembled only from saved numbers"],
                    ["metrics_test_original.csv, metrics_test_full.csv", "accuracy, macro precision, recall, F1 with 95 % intervals"],
                    ["per_class_f1_*.csv", "F1 of each of the six classes for both models"],
                    ["targets_check.csv", "accuracy >= 90 % and macro-F1 > 0.85: PASS/FAIL per model and view"],
                    ["efficiency.csv", "parameters, file size, latency, training time"],
                    ["training_curves.png, metric_bars_*.png", "training/validation curves; metric bars with intervals and target lines"],
                    ["final_model.json", "which model the application will use, and why"],
                    ["workspace/runs/&lt;run&gt;/seed42/eval/", "reports (precision/recall/F1 per class), confusion matrices (counts and normalised), per-image predictions"],
                    ["results/error_analysis/&lt;run&gt;/", "most frequent mistakes, mistakes gallery, Grad-CAM, written error report"]],
                   [0.38, 0.62])
    story += [H2("8.1 How to read the results"),
              *bullets(["Compare the two models on the <b>same</b> view. The primary view (test originals) is the headline; the full view shows robustness to rotated/noisy copies.",
                        "The test set contains a few hundred independent photos, so differences of one or two points may be noise: look at the 95 % intervals and the McNemar p-value in the report (p above 0.05 = could be chance).",
                        "<b>If a target is missed, report it honestly</b> and use the error analysis (top confusions, fresh-vs-rotten vs fruit-type mistakes, Grad-CAM) to explain why.",
                        "<b>If accuracy is 99 % or higher, be suspicious</b> rather than proud: re-read the leakage audit and the split verification output.",
                        "Which model suits limited hardware? Use the efficiency table and the optional TensorFlow Lite export: <font name='DVM'>python scripts/export_tflite.py --run mobilenet_v2</font>."])]

    # ---------------- 9 app
    story += [PageBreak(), H1("9. Phase E - the application")]
    story += callout("RUN ON", "<b>Your laptop</b> (recommended for the demonstration) or the DGX. File: <b>app/app.py</b>. Needs <i>workspace/runs/...</i> and <i>results/final_model.json</i> (from <i>dgx_results.zip</i>).")
    story += [H2("9.1 Bring the results back and install"),
              code("# on your laptop\nscp user@dgx-host:path/fresh-rotten-fruit-capstone/dgx_results.zip .\nunzip -o dgx_results.zip -d fresh-rotten-fruit-capstone\ncd fresh-rotten-fruit-capstone\npip install -r requirements.txt"),
              P("Use the same major TensorFlow/Keras version as on the DGX (Keras 3) so the <i>.keras</i> file loads.")]
    story += [H2("9.2 Launch"),
              code("python app/app.py                                   # final model chosen by compare.py\npython app/app.py --model workspace/runs/mobilenet_v2/seed42/model.keras\npython app/app.py --model results/export/mobilenet_v2.tflite   # lightweight TFLite model\npython app/app.py --host 0.0.0.0 --share             # reachable from other machines / temporary public link"),
              P("Open <b>http://127.0.0.1:7860</b> in the browser. Upload a fruit photo and press <b>Classify</b>. The page shows the fruit, <b>FRESH or ROTTEN</b>, the predicted class, "
                "the confidence of that class, the fresh/rotten confidence, a bar chart of all six probabilities, and a warning when the confidence is below 60 %. "
                "Tick the Grad-CAM box (Keras models) to see where the model looked.")]
    story += figure("fig4_app.png", "Figure 4 - Layout of the application (schematic; values are placeholders).", 0.85)
    story += [H2("9.3 Running the app on the DGX"),
              code("python app/app.py --host 127.0.0.1 --port 7860        # on the DGX\nssh -L 7860:localhost:7860 user@dgx-host                    # on your laptop, second terminal\n# then open http://127.0.0.1:7860 on your laptop"),
              H2("9.4 Command line and notebook alternatives"),
              code("python scripts/predict.py --model auto --image apple.jpg banana.jpg          # prints class | FRESH/ROTTEN | confidence\npython scripts/predict.py --model auto --image apple.jpg --json                   # machine-readable"),
              P("Notebook 04 contains the same pipeline plus the real-world photo test: put your own phone photos in <font name='DVM'>real_world/&lt;class_name&gt;/*.jpg</font> "
                "(class names: freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges). These photos must never be used for training.")]
    story += callout("TIP", "Before the demonstration: run the app once with a fresh photo and a rotten photo of each fruit, and keep a backup of <i>dgx_results.zip</i>.")

    # ---------------- 10 troubleshooting
    story += [PageBreak(), H1("10. Troubleshooting guide")]
    trouble = [["Symptom", "Likely cause", "Fix"],
               ["Notebook: <i>Project folder not found</i>", "zip not uploaded to /content", "upload fresh-rotten-fruit-capstone.zip in the Colab Files panel (or set REPO_URL) and re-run the SETUP cell"],
               ["Kaggle download fails", "no internet / rate limit", "retry; or download the dataset in a browser, unzip, set FRUIT_DATA_ROOT"],
               ["<i>ModuleNotFoundError: fruitfresh</i>", "wrong folder or SETUP cell skipped", "run commands from the project root; run the SETUP cell first"],
               ["verify_data: <i>images missing</i>", "bundle not unpacked inside the project, wrong --data-root", "unzip in the project folder; check data/raw/train/&lt;class&gt;/ exists"],
               ["<i>test.csv changed since the split was created</i>", "CSV edited or re-saved", "re-copy CSVs + split_manifest.json from the bundle; never open them in a spreadsheet"],
               ["check_gpu: no GPU visible", "wrong environment, driver/CUDA mismatch", "nvidia-smi; use a Blackwell-capable TensorFlow build or NGC container (section 7.1)"],
               ["<i>no kernel image is available</i>", "TensorFlow build lacks B200 kernels", "use a TensorFlow/NGC release that supports Blackwell (CUDA 12.8+)"],
               ["Out of memory", "batch too large / GPU shared", "--per-gpu-batch-size 16; check nvidia-smi for other users"],
               ["Hang or NCCL error at start of multi-GPU training", "GPU communication", "--config configs/single_gpu_debug.json to isolate; NCCL_DEBUG=INFO; ask the administrators"],
               ["Training slow, GPUs idle", "data on slow disk / few CPU cores", "copy data to local SSD; allocate more CPU cores; epoch 1 is slower (images get cached)"],
               ["MobileNetV2: weights download fails", "DGX offline", "fetch_pretrained_weights.py elsewhere; copy assets/*.h5; use --weights-file"],
               ["Accuracy around 17 % and flat", "wrong labels/paths, bad settings", "check train_info.json and history curves; retrain with --force"],
               ["Test accuracy 99 % or more", "possible leakage / easy test set", "read the leakage audit; do not present as success without analysis"],
               ["One class has very low recall", "similar classes (e.g. rotten vs fresh orange), few test blocks", "error_analysis.py: top confusions and Grad-CAM; consider deeper fine-tuning"],
               ["compare.py --use-best fails", "results/best_*.json missing", "run summarize_experiments.py for both families first"],
               ["App: model/final_model.json not found", "compare.py not run or results not copied", "run compare.py or pass --model &lt;path&gt;"],
               ["App: port in use", "another Gradio instance", "python app/app.py --port 7861"],
               ["<i>.keras</i> file will not load", "different Keras version", "use TensorFlow 2.16 or newer (Keras 3) on both machines"],
               ["Colab runtime reset, files gone", "Colab disk is temporary", "re-run SETUP; keep the downloaded bundle on your computer; for runs use backup_workspace / restore_workspace"]]
    story += table(trouble, [0.27, 0.27, 0.46])

    # ---------------- 11 checklist
    story += [PageBreak(), H1("11. Final verification checklist")]
    chk = [["", "Check", "How to verify"],
           ["☐", "Dataset downloaded and explored", "notebook 01 sections 1-3 printed without errors; unreadable images = 0"],
           ["☐", "Split created and verified", "verify_split passed; minimum gap at least 60 s; counts close to 70/15/15"],
           ["☐", "Bundle on the DGX is intact", "verify_data.py prints DATA OK"],
           ["☐", "All GPUs usable", "check_gpu.py prints GPU CHECK PASSED"],
           ["☐", "Model 1 trained", "workspace/runs/baseline_cnn/seed42/model.keras, history.csv, train_info.json exist"],
           ["☐", "Model 2 trained in two stages", "stage1.keras and model.keras exist; train_info.json: finetune stage, backbone_batchnorm_trainable = 0"],
           ["☐", "Training curves look sensible", "results/training_curves.png: validation loss falls; no huge gap"],
           ["☐", "Both models evaluated on the same data", "eval/metrics.json in both runs; fairness table identical except the model"],
           ["☐", "Accuracy, precision, recall, F1 reported", "results/metrics_*.csv, per_class_f1_*.csv, report_*.txt"],
           ["☐", "Confusion matrices saved", "eval/confusion_*.png (counts and normalised)"],
           ["☐", "Proposal targets checked", "results/targets_check.csv shows PASS or FAIL honestly for each model"],
           ["☐", "Comparison report generated", "results/comparison_report.md incl. McNemar test and efficiency table"],
           ["☐", "Error analysis done", "results/error_analysis/<run>/ has the report, gallery and Grad-CAM"],
           ["☐", "Final model chosen by the fixed rule", "results/final_model.json"],
           ["☐", "Application runs and gives correct output format", "upload a photo: class, FRESH/ROTTEN, confidence shown"],
           ["☐", "Real-world photos tested (recommended)", "notebook 04, section 4"],
           ["☐", "Results copied back and backed up", "dgx_results.zip on your laptop"],
           ["☐", "Automated tests pass", "python -m pytest tests -q   (about 10 minutes, CPU)"],
           ["☐", "No result in the report is typed by hand", "every number comes from results/ or train_info.json"]]
    story += table(chk, [0.05, 0.38, 0.57])

    # ---------------- 12 testing status
    story += [H1("12. What was tested, and what needs the college GPU server")]
    summary = ROOT / "docs" / "test_summary.json"
    if summary.exists():
        import json
        t = json.loads(summary.read_text())
        stext = (f"Last full run ({t['date']}, {t['environment']}, Python {t['python']}, TensorFlow {t['tensorflow']}):\n"
                 f"  automated tests : {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped (of {t['tests']}), {t['pytest_seconds']} s\n"
                 f"  notebooks       : {t.get('notebooks_executed', '?')} of {t.get('notebooks_total', '?')} executed top to bottom (reduced mode)")
    else:
        stext = "No test summary found: run  python tools/run_tests.py  to create docs/test_summary.json"
    story += [P("Tested in the development environment (CPU only, a <b>synthetic</b> dataset with the same folder layout and file naming as the Kaggle data):"),
              *bullets(["data indexing, file-name parsing, exact-duplicate removal, time-block / random / cross-validation splits and their leakage checks",
                        "split CSV hashing, data bundle creation and verification",
                        "both model architectures, the freezing logic (BatchNorm stays frozen), training of both models end to end, resume after interruption",
                        "multi-GPU training code with <b>two logical CPU replicas</b> (MirroredStrategy)",
                        "evaluation metrics, bootstrap intervals, McNemar test, comparison report, error analysis, Grad-CAM",
                        "TensorFlow Lite export and agreement with the Keras model",
                        "inference pipeline, command-line prediction, and the <b>real Gradio server</b> called through its API (with and without Grad-CAM)",
                        "the one-command DGX shell pipeline (with CPU allowed) and the results package layout",
                        "all four notebooks executed top to bottom in a reduced mode (1 epoch, 64 px images, no ImageNet weights)",
                        "every command and option written in the README/docs exists in the scripts (automated documentation test)"]),
              code(stext)]
    story += [P("<b>Needs your environment (not tested here):</b>")]
    story += table([["Item", "Why it could not be tested", "How you will know it works"],
                    ["Real Kaggle download and the real dataset statistics", "no access to Kaggle from the development environment", "notebook 01 outputs (counts, audit)"],
                    ["Running on Google Colab", "no Colab access", "notebook 01 runs to <i>READY FOR THE DGX</i>"],
                    ["DGX B200 execution, NCCL multi-GPU, Blackwell compatibility, bfloat16", "no GPU server", "check_gpu.py passes; training prints the hardware line and finishes"],
                    ["ImageNet weight download", "blocked in the test environment", "MobileNetV2 stage 1 starts (or use the offline weights file)"],
                    ["All accuracy / F1 numbers and the PASS/FAIL outcome", "they exist only after training on real data", "scripts/compare.py output"],
                    ["Training time and GPU speed-up", "not measured", "train_info.json: training_seconds"]],
                   [0.35, 0.30, 0.35])

    # ---------------- appendix
    story += [PageBreak(), H1("Appendix A - Command-line reference (generated from the scripts)")]
    story += [P("The text below is the real <font name='DVM'>--help</font> output of each script, captured when this manual was built.")]
    scripts = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("_")) + ["../app/app.py"]
    for s in scripts:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s), "--help"], capture_output=True, text=True, timeout=180,
                           env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin:/usr/local/bin", "COLUMNS": "100"})
        name = "app/app.py" if s.endswith("app.py") else f"scripts/{s}"
        lines = [l[:108] for l in r.stdout.strip().splitlines()]
        story += [KeepTogether([P(f"<b>{name}</b>", "H3"), code("\n".join(lines[:42]))])]
    story += [H1("Appendix B - Glossary")]
    story += table([["Term", "Meaning"],
                    ["Source photo", "one original photo; its augmented copies share its identity and always stay in the same split"],
                    ["Time-block split", "splitting by capture time in contiguous blocks, with a time buffer between splits"],
                    ["Transfer learning", "starting from a network pre-trained on ImageNet and adapting it to our six classes"],
                    ["Fine-tuning", "training some of the pre-trained layers with a very small learning rate"],
                    ["MirroredStrategy", "TensorFlow's way to train on several GPUs at once with identical copies of the model"],
                    ["Global batch size", "images per training step across all GPUs = per-GPU batch x number of GPUs"],
                    ["Macro-F1", "average of the F1-scores of the six classes (each class counts equally)"],
                    ["Confidence interval (95 %)", "range in which the true value probably lies, computed by resampling source photos"],
                    ["McNemar test", "tests whether two models differ on the same images, looking only at images where one is right and the other wrong"],
                    ["Grad-CAM", "heat-map showing which image regions influenced the prediction"],
                    ["TFLite", "compact model format for phones and small devices"]], [0.25, 0.75])
    doc.multiBuild(story)
    return OUT


if __name__ == "__main__":
    fig_architecture(); fig_models(); fig_split(); fig_app()
    out = build_pdf()
    print("written:", out, f"({out.stat().st_size / 1e3:.0f} KB)")
