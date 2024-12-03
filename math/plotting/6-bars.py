#!/usr/bin/env python3
"""Plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar graph
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    bar_width = 0.5

    x = np.arange(len(people))

    plt.bar(x, fruit[0], width=bar_width, label="apples", color=colors[0])
    plt.bar(x, fruit[1], width=bar_width,
            bottom=fruit[0], label="bananas", color=colors[1])
    plt.bar(x, fruit[2], width=bar_width,
            bottom=fruit[0] + fruit[1], label="oranges", color=colors[2])
    plt.bar(x, fruit[3], width=bar_width,
            bottom=fruit[0] + fruit[1] + fruit[2],
            label="peaches", color=colors[3])

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(range(0, 81, 10))
    plt.legend(loc="upper right")

    plt.show()
