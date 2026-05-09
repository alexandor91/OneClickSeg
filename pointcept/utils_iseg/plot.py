import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_curves(
    x,
    y_dict_list,
    color_map,
    linestyle_map,
    setting_map,
):
    """"""
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.linewidth": 3,
            "lines.markersize": 12,
        }
    )
    plt.figure(figsize=(8, 6))
    for y_dict in y_dict_list:
        # color marker 共用一个名称
        color, marker = color_map[y_dict["color_name"]]
        linestyle = linestyle_map[y_dict["linestyle_name"]]
        plt.plot(
            x,
            y_dict["y"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            alpha=0.8,
        )

    # 构造 legend 句柄
    color_title = color_map.pop("title")
    handles = [Line2D([], [], color="none", label=color_title)]
    labels = [color_title]
    for color_name, (color, marker) in color_map.items():
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                label=color_name,
            )
        )
        labels.append(color_name)
    # 优先使用color作为区分, 如果 linestyle 只有一种就不列出
    linestyle_title = linestyle_map.pop("title")
    if len(linestyle_map) > 1:
        handles.append(Line2D([], [], color="none", label=linestyle_title))
        labels.append(linestyle_title)
        for linestyle_name, linestyle_value in linestyle_map.items():
            handles.append(
                Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=linestyle_value,
                    label=linestyle_name,
                )
            )
            labels.append(linestyle_name)
    plt.legend(handles=handles, labels=labels, loc="best", frameon=True)
    plt.xlabel(setting_map["xlabel"])
    plt.ylabel(setting_map["ylabel"])
    plt.title(setting_map["title"])
    plt.grid(True, linestyle=":", alpha=0.8)
    plt.tight_layout()
    if "save_path" in setting_map:
        save_path = setting_map["save_path"]
        plt.savefig(save_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    color_map = {
        "title": "experiment",
        "50 trained queries": ["#1f77b4", "o"],  # 蓝色
        "100 trained queries": ["#ff7f0e", "^"],  # 橙色
        "200 trained queries": ["#2ca02c", "s"],  # 绿色
    }
    linestyle_map = {
        "title": "",
        "all": "-",
        # '200 trained queries': '--',
    }
