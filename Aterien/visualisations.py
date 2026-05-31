import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.dates as mdates
def configure_axis(ax: Axes, df_name: pd.DataFrame) -> None:
    """Configures the X-axis for date formatting and grid."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlim(min(df_name["date"]), max(df_name["date"]))
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.grid(True,color='#D5D5D5')


def draw_event_line(ax: Axes, event_dict: dict) -> None:
    """
    Draws vertical dashed lines for significant events and adds a label at the top.
    Uses ax.get_xaxis_transform() to keep Y-coordinates relative (0 to 1)
    while X-coordinates remain as dates.
    """
    if not event_dict:
        return

    for event_name, date_str in event_dict.items():
        date_obj = pd.to_datetime(date_str)
        ax.axvline(x=date_obj, color='dimgray', linestyle='--', alpha=0.7, linewidth=1.5)

        # Placing event name on the plot. Y = 0.98 means 98% of plot hight
        ax.text(date_obj, 0.98, f' {event_name}', color='dimgray', rotation=90,
                transform=ax.get_xaxis_transform(), va='top', ha='right', fontsize=9)


def simple_descriptive_plots_grid(df: pd.DataFrame,color_dict:dict, event_dict: dict = None, figname: str = None) -> None:
    """
    Creates a 2x2 grid dashboard.
    Top row: Epidemic data (Cases vs Deaths).
    Bottom row: Vaccination data (Daily absolute vs Cumulative % of population).
    Legends are placed in the upper left corner of every subplot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # ========================================================
    # [0, 0] DAILY CASES & DEATHS
    # ========================================================
    ax00_left = axs[0, 0]
    ax00_right = ax00_left.twinx()

    ax00_left.set_title("Daily Epidemic Dynamics")
    ax00_left.set_ylabel("Daily Cases", color=color_dict["CONFIRMED"])
    ax00_right.set_ylabel("Daily Deaths", color=color_dict["DECEASED"])

    ax00_left.scatter(df['date'], df['new_confirmed'], color=color_dict["CONFIRMED"], alpha=0.15, s=10, label="Daily Confirmed")
    ax00_left.plot(df['date'], df['new_confirmed'].rolling(7).mean(), color=color_dict["CONFIRMED"], linewidth=2, label="Confirmed (7d avg)")

    ax00_right.scatter(df['date'], df['new_deceased'], color=color_dict["DECEASED"], alpha=0.15, s=10, label="Daily Deaths")
    ax00_right.plot(df['date'], df['new_deceased'].rolling(7).mean(), color=color_dict["DECEASED"], linewidth=2, label="Deaths (7d avg)")


    draw_event_line(ax00_left, event_dict)
    configure_axis(ax00_left, df)

    # Multiple legends for each axis
    ax00_left.legend(loc='upper left', title="Left axis (Confirmed cases)")
    ax00_right.legend(loc='upper right',title="Right axis (Deaths)")

    # ========================================================
    # [0, 1] CUMULATIVE CASES & DEATHS
    # ========================================================
    ax01_left = axs[0, 1]
    ax01_right = ax01_left.twinx()

    ax01_left.set_title("Cumulative Epidemic Totals")
    ax01_left.set_ylabel("Total Cases", color=color_dict["CONFIRMED"])
    ax01_right.set_ylabel("Total Deaths", color=color_dict["DECEASED"])

    ax01_left.plot(df['date'], df['cumulative_confirmed'], color=color_dict["CONFIRMED"], linewidth=2, label="Cum. Cases")
    ax01_right.plot(df['date'], df['cumulative_deceased'], color=color_dict["DECEASED"], linewidth=2, label="Cum. Deaths")

    draw_event_line(ax01_left, event_dict)
    configure_axis(ax01_left, df)

    # Legends for each axis
    ax01_left.legend(loc='upper left', title="Left axis (Cum. confirmed cases)")
    ax01_right.legend(loc='lower right',title="Right axis (Deaths)")

    # ========================================================
    # [1, 0] DAILY VACCINATIONS
    # ========================================================
    ax10 = axs[1, 0]
    ax10.set_title("Daily Vaccination Progress")
    ax10.set_ylabel("Doses Given Per Day")

    ax10.scatter(df['date'], df['new_persons_vaccinated'], color=color_dict["VACCINATED"], alpha=0.15, s=10,label="First Dose")
    ax10.plot(df['date'], df['new_persons_vaccinated'].rolling(7).mean(), color=color_dict["VACCINATED"], linewidth=2, label="1st Dose (7d avg)")

    ax10.scatter(df['date'], df['new_persons_fully_vaccinated'], color=color_dict["FULLY_VACCINATED"], alpha=0.15, s=10,label="Fully Vacc.")
    ax10.plot(df['date'], df['new_persons_fully_vaccinated'].rolling(7).mean(), color=color_dict["FULLY_VACCINATED"], linewidth=2, label="Fully Vacc. (7d avg)")

    draw_event_line(ax10, event_dict)
    configure_axis(ax10, df)
    ax10.legend(loc='upper left')

    # ========================================================
    # [1, 1] CUMULATIVE VACCINATIONS (% of Population)
    # ========================================================
    ax11 = axs[1, 1]
    ax11.set_title("Cumulative Vaccination (% of Population)")
    ax11.set_ylabel("Percentage of Population (%)")

    perc_vacc = (df['cumulative_persons_vaccinated'] / df['population']) * 100
    perc_full = (df['cumulative_persons_fully_vaccinated'] / df['population']) * 100

    ax11.plot(df['date'], perc_vacc, color=color_dict["VACCINATED"], linewidth=2, label="Cum. 1st Dose (%)")
    ax11.plot(df['date'], perc_full, color=color_dict["FULLY_VACCINATED"], linewidth=2, label="Cum. Fully Vacc. (%)")

    draw_event_line(ax11, event_dict)
    configure_axis(ax11, df)
    ax11.legend(loc='lower right')

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, bbox_inches='tight')
    plt.show()


def cumulative_totals_plot(df: pd.DataFrame,color_dict:dict, event_dict: dict = None, figname: str = None) -> None:
    """
    Creates a standalone plot for overall cumulative totals with external legends.
    """
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()

    ax_left.set_title("Overall Cumulative Totals: Cases, Vaccinations, and Deaths")
    ax_left.set_ylabel("Total Cases / Vaccinations")
    ax_right.set_ylabel("Total Deaths", color=color_dict["DECEASED"])

    ax_left.plot(df['date'], df['cumulative_confirmed'], color=color_dict["CONFIRMED"], linewidth=2, label='Cum. Confirmed')
    ax_left.plot(df['date'], df['cumulative_persons_vaccinated'], color=color_dict["VACCINATED"], linewidth=2, label='Cum. Vaccinated')
    ax_left.plot(df['date'], df['cumulative_persons_fully_vaccinated'], color=color_dict["FULLY_VACCINATED"], linewidth=2, label='Cum. Fully Vacc.')

    ax_right.plot(df['date'], df['cumulative_deceased'], color=color_dict["DECEASED"], linewidth=2, label='Cum. Deaths')

    draw_event_line(ax_left, event_dict)
    configure_axis(ax_left, df)

    ax_left.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Left Axis")
    ax_right.legend(loc='lower left', bbox_to_anchor=(1.05, 0.6), title="Right Axis")

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, bbox_inches='tight')
    plt.show()


def cfr_plot(df: pd.DataFrame,color_dict:dict, event_dict: dict = None, figname: str = None) -> None:
    """
    Plots the 14-day lagged Case Fatality Rate (CFR) using 7-day rolling averages.
    Formula: CFR_t = (Deaths_t / Cases_{t-14}) * 100
    Applies event lines to the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Calculate 7-day rolling averages to smooth out weekly noise
    smooth_cases = df['new_confirmed'].rolling(7).mean()
    smooth_deaths = df['new_deceased'].rolling(7).mean()

    # 2. Calculate CFR with a 14-day lag for cases
    # shift(14) moves cases forward in time, so deaths at day T are divided by cases at day T-14
    lagged_cases = smooth_cases.shift(14)

    # Avoid division by zero and handle infinity values if cases were 0
    cfr_series = (smooth_deaths / lagged_cases) * 100
    cfr_series = cfr_series.replace([np.inf, -np.inf], np.nan)

    # 3. Plotting
    ax.plot(df['date'], cfr_series, color=color_dict["CFR"], linewidth=2.5, label='Lagged CFR (7d avg)')

    # Set labels and titles
    ax.set_title("Lagged Case Fatality Rate (CFR) Dynamics")
    ax.set_ylabel("Case Fatality Rate (%)")
    ax.set_xlabel("Date")

    # 4. Apply external helpers (grid, formatting, and event lines)
    draw_event_line(ax, event_dict)
    configure_axis(ax, df)

    ax.legend(loc='upper right')

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, bbox_inches='tight')
    plt.show()