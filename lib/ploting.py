import seaborn as sns
from matplotlib.transforms import BboxBase
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_style(font_scale: float = 0.8) -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=font_scale,
        color_codes=True,
    )
     # Matplotlib settings
    plt.rc("axes", linewidth=0.4)
    plt.rc("grid", linewidth=0.4)
    plt.rc("hatch", linewidth=0.5)
    plt.rc("lines", linewidth=1.0)
    plt.rc("patch", linewidth=0.5)
    plt.rc("xtick.major", width=0.4)
    plt.rc("xtick.minor", width=0.3)
    plt.rc("ytick.major", width=0.4)
    plt.rc("ytick.minor", width=0.3)
    plt.rc("font", family="Computer Modern")

    # Additional Matplotlib settings for LaTeX rendering
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\newcommand{\system}[0]{\textsc{Tbd}}"

def plot_ylabel(path: str, label: str, bbox: BboxBase, height: float, width: float = 0.25) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3, height))
    fig.supylabel(label)
    ax.remove()
    plt.tight_layout(pad=1.02)
    label_bbox = fig.get_tightbbox()
    adjusted_bbox = Bbox(((label_bbox.x0, bbox.y0), (label_bbox.x1, bbox.y1 + 0.1)))
    plt.savefig(path, bbox_inches=adjusted_bbox)
    plt.close()

def plot_legend(path: str, handles: list[Artist], labels: list[str], ncol: int = 1) -> None:
    fig = plt.figure(figsize=(10, 3))

    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        columnspacing=2,
        borderaxespad=0.5,
        labelspacing=0.5,
        fontsize=mpl.rcParams["font.size"],
        handlelength=1.5,
    )

    plt.tight_layout(pad=1.02)
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()
    plt.close()