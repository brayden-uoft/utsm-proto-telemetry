"""Render the physics and optimization diagrams used by the Front Campus model docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"
GREEN = "#16a34a"
GREEN_LIGHT = "#dcfce7"
ORANGE = "#ea580c"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_LIGHT = "#ede9fe"
RED = "#dc2626"
SLATE = "#334155"
SLATE_LIGHT = "#f1f5f9"
GRID = "#cbd5e1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for the two generated PNG files",
    )
    return parser.parse_args()


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = GRID,
    fontsize: float = 11,
    color: str = SLATE,
    linewidth: float = 1.5,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        linespacing=1.35,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SLATE,
    linewidth: float = 2.0,
    mutation_scale: float = 16,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
        )
    )


def setup_canvas(width: float = 16, height: float = 9) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def draw_car(ax: plt.Axes) -> None:
    ax.plot([0.55, 5.85], [2.55, 3.10], color=SLATE, linewidth=3)
    body = FancyBboxPatch(
        (1.75, 3.25),
        2.7,
        0.75,
        boxstyle="round,pad=0.05,rounding_size=0.16",
        facecolor=BLUE,
        edgecolor="#1e3a8a",
        linewidth=2,
    )
    cab = FancyBboxPatch(
        (2.45, 3.87),
        1.25,
        0.55,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=BLUE_LIGHT,
        edgecolor="#1e3a8a",
        linewidth=2,
    )
    ax.add_patch(body)
    ax.add_patch(cab)
    for center in ((2.25, 3.18), (3.95, 3.18)):
        ax.add_patch(Circle(center, 0.28, facecolor=SLATE, edgecolor="#0f172a"))
        ax.add_patch(Circle(center, 0.10, facecolor="#94a3b8", edgecolor="none"))

    add_arrow(ax, (4.55, 3.78), (6.15, 3.95), color=GREEN, linewidth=3)
    ax.text(5.32, 4.20, "required propulsion", ha="center", color=GREEN, fontsize=11, weight="bold")
    add_arrow(ax, (1.70, 3.67), (0.30, 3.52), color=RED, linewidth=3)
    ax.text(0.95, 3.90, "resistance", ha="center", color=RED, fontsize=11, weight="bold")


def write_physics_forces(output: Path) -> None:
    fig, ax = setup_canvas()
    ax.text(
        0.4,
        8.55,
        "Physics floor for each track section",
        fontsize=23,
        weight="bold",
        color="#0f172a",
        va="top",
    )
    ax.text(
        0.4,
        8.05,
        "The simulation converts a speed change into time, acceleration, force, power, and energy.",
        fontsize=12.5,
        color="#475569",
        va="top",
    )

    draw_car(ax)
    ax.text(3.1, 5.25, "What the car must overcome", ha="center", fontsize=14, weight="bold", color=SLATE)
    force_rows = [
        ("rolling", r"$F_{roll}=m g C_{rr}$"),
        ("grade", r"$F_{grade}=m g\,(grade/100)$"),
        ("acceleration", r"$F_{accel}=m\,max(a,0)$"),
        ("air drag", r"$F_{aero}=\frac{1}{2}\rho C_d A v^2$"),
        ("corner scrub", r"$F_{corner}=k_c m v^2\,max(\kappa,0)$"),
    ]
    y = 1.95
    for index, (name, formula) in enumerate(force_rows):
        col = index % 2
        row = index // 2
        x = 0.45 + col * 2.95
        yy = y - row * 0.68
        add_box(
            ax,
            (x, yy),
            2.65,
            0.46,
            f"{name}:  {formula}",
            facecolor=SLATE_LIGHT,
            edgecolor=GRID,
            fontsize=9.8,
        )

    ax.plot([6.45, 6.45], [0.55, 7.55], color=GRID, linewidth=1.4)
    ax.text(7.0, 7.35, "1. Speed and section geometry", fontsize=14, weight="bold", color=BLUE)
    add_box(
        ax,
        (7.0, 6.10),
        8.15,
        0.85,
        r"$v=v_{kph}/3.6$     $a=(v_1^2-v_0^2)/(2s)$     $t=s/((v_0+v_1)/2)$",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        fontsize=13,
    )
    ax.text(
        7.15,
        5.82,
        "Each approximately 20 m section uses its entry speed, target speed, grade, and curvature.",
        fontsize=10.5,
        color="#475569",
    )

    ax.text(7.0, 5.25, "2. Required propulsion", fontsize=14, weight="bold", color=PURPLE)
    add_box(
        ax,
        (7.0, 4.00),
        8.15,
        0.85,
        r"$F_{req}=max(F_{roll}+F_{grade}+F_{accel}+F_{aero}+F_{corner},\ 0)$",
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        fontsize=12.5,
    )

    ax.text(7.0, 3.43, "3. Electrical power and energy floor", fontsize=14, weight="bold", color=GREEN)
    add_box(
        ax,
        (7.0, 2.18),
        8.15,
        0.85,
        r"$P_{physics}=F_{req}v/\eta$        $E_{section}=P_{model}t$",
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        fontsize=14,
    )
    ax.text(
        7.15,
        1.85,
        "The learned model cannot predict less propulsion power than this physics result.",
        fontsize=10.5,
        color="#475569",
    )
    add_box(
        ax,
        (7.0, 0.62),
        8.15,
        0.73,
        "Coast is explicit: motor propulsion current and power are set to zero.",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        fontsize=11,
        color="#9a3412",
    )

    ax.text(
        0.45,
        0.22,
        "Defaults: total mass 100 kg, Crr 0.008, drivetrain efficiency 0.82, CdA 0.07235 m2, air density 1.225 kg/m3, corner factor 0.1",
        fontsize=9.5,
        color="#64748b",
    )
    fig.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_optimizer_lattice(ax: plt.Axes) -> None:
    x_values = [11.0, 12.0, 13.0, 14.0, 15.0]
    y_values = [3.55, 4.05, 4.55, 5.05]
    chosen = [1, 2, 2, 3, 2]
    for col in range(len(x_values) - 1):
        for row in range(len(y_values)):
            for next_row in range(max(0, row - 1), min(len(y_values), row + 2)):
                ax.plot(
                    [x_values[col], x_values[col + 1]],
                    [y_values[row], y_values[next_row]],
                    color="#dbe3ec",
                    linewidth=0.8,
                    zorder=1,
                )
    for col, x in enumerate(x_values):
        for row, y in enumerate(y_values):
            active = row == chosen[col]
            ax.add_patch(
                Circle(
                    (x, y),
                    0.105 if active else 0.075,
                    facecolor=ORANGE if active else "white",
                    edgecolor=ORANGE if active else "#94a3b8",
                    linewidth=1.5,
                    zorder=3,
                )
            )
    for col in range(len(x_values) - 1):
        ax.plot(
            [x_values[col], x_values[col + 1]],
            [y_values[chosen[col]], y_values[chosen[col + 1]]],
            color=ORANGE,
            linewidth=3.2,
            zorder=2,
        )
    ax.text(13.0, 5.43, "candidate speed states", ha="center", fontsize=10, color="#64748b")
    ax.text(13.0, 3.15, "track sections", ha="center", fontsize=10, color="#64748b")
    ax.text(10.55, 4.30, "speed", ha="center", va="center", rotation=90, fontsize=10, color="#64748b")


def write_optimization_math(output: Path) -> None:
    fig, ax = setup_canvas()
    ax.text(
        0.4,
        8.55,
        "From Front Campus data to an efficient lap",
        fontsize=23,
        weight="bold",
        color="#0f172a",
        va="top",
    )
    ax.text(
        0.4,
        8.05,
        "A weighted regression learns the car. A constrained search chooses the speed path.",
        fontsize=12.5,
        color="#475569",
        va="top",
    )

    column_titles = [
        (0.45, "1. Learn from 15 laps", BLUE),
        (5.65, "2. Combine data and physics", PURPLE),
        (10.85, "3. Search the lap", ORANGE),
    ]
    for x, title, color in column_titles:
        ax.text(x, 7.35, title, fontsize=14, weight="bold", color=color)
    ax.plot([5.25, 5.25], [0.45, 7.55], color=GRID, linewidth=1.2)
    ax.plot([10.45, 10.45], [0.45, 7.55], color=GRID, linewidth=1.2)

    add_box(
        ax,
        (0.55, 5.83),
        4.05,
        0.92,
        "3 runs, 15 laps, 8,115 matched samples\nEach lap has equal total weight",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        fontsize=11.2,
    )
    add_arrow(ax, (2.58, 5.80), (2.58, 5.40), color=BLUE)
    add_box(
        ax,
        (0.55, 4.15),
        4.05,
        1.12,
        "Features x\nspeed, speed2, accel, braking, grade,\naction flags, accel x speed",
        facecolor="white",
        edgecolor=GRID,
        fontsize=10.6,
    )
    add_arrow(ax, (2.58, 4.10), (2.58, 3.72), color=BLUE)
    add_box(
        ax,
        (0.55, 2.40),
        4.05,
        1.15,
        r"$\beta=(X^T W X+\lambda R)^{-1}X^T W y$" + "\nweighted ridge regression, ridge = 0.001",
        facecolor=SLATE_LIGHT,
        edgecolor=SLATE,
        fontsize=11.2,
    )
    add_arrow(ax, (2.58, 2.36), (2.58, 1.98), color=BLUE)
    add_box(
        ax,
        (0.55, 0.83),
        4.05,
        0.98,
        r"$I_{ridge}=x\beta_I$       $P_{ridge}=x\beta_P$" + "\nposition is set to zero for track transfer",
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        fontsize=11.2,
    )

    add_box(
        ax,
        (5.75, 5.67),
        4.05,
        1.08,
        "Driving action\naccelerate, hold, or coast",
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        fontsize=11.5,
    )
    add_arrow(ax, (7.78, 5.62), (7.78, 5.24), color=PURPLE)
    add_box(
        ax,
        (5.75, 3.83),
        4.05,
        1.22,
        r"$I_{model}=max(I_{duty},\ P_{physics}/V)$" + "\n" + r"$P_{model}=max(P_{ridge},\ V I_{model},\ P_{physics})$",
        facecolor="white",
        edgecolor=PURPLE,
        fontsize=11.7,
    )
    add_arrow(ax, (7.78, 3.78), (7.78, 3.40), color=PURPLE)
    add_box(
        ax,
        (5.75, 2.13),
        4.05,
        1.08,
        r"$E_i=P_{model,i}\,t_i$" + "\nenergy is summed across the lap",
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        fontsize=12.5,
    )
    add_box(
        ax,
        (5.75, 0.72),
        4.05,
        0.80,
        "Coast sets propulsion to 0 W\nbefore the hybrid calculation",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        fontsize=10.8,
        color="#9a3412",
    )

    draw_optimizer_lattice(ax)
    add_box(
        ax,
        (10.75, 1.73),
        4.75,
        1.12,
        r"$C=\sum_i(E_i+\lambda_t t_i+penalty(I_{peak,i}))$" + "\nchoose the lowest-cost connected path",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        fontsize=11.2,
    )
    add_box(
        ax,
        (10.75, 0.47),
        4.75,
        0.88,
        "Constraints: 58.2 to 60.0 s lap, 20 A fuse,\n1.0 s burst, speed-step, speed-change, closed loop",
        facecolor=SLATE_LIGHT,
        edgecolor=SLATE,
        fontsize=10.4,
    )
    ax.text(
        13.12,
        6.05,
        "Dynamic programming keeps the best\nway to reach each speed and fuse state.",
        ha="center",
        fontsize=10.8,
        color="#475569",
        linespacing=1.35,
    )
    fig.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    write_physics_forces(output_dir / "front-campus-physics-forces.png")
    write_optimization_math(output_dir / "front-campus-optimization-math.png")
    print(output_dir / "front-campus-physics-forces.png")
    print(output_dir / "front-campus-optimization-math.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
