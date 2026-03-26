#!/usr/bin/env python3
# plot.py — Plot simulation results against paper's figures
# Run after: ./sim

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_x_positions():
    """Fig 3/5: Agent x-position actual vs demanded over time"""
    df = pd.read_csv('x_positions.csv')

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('X-axis position: actual vs demanded (cf. Paper Fig 3/5)', fontsize=14)

    for i in range(5):
        ax = axes[i]
        ax.plot(df['time'], df[f'drone{i}_x'], label='actual', linewidth=1)
        ax.plot(df['time'], df[f'drone{i}_x_desired'], '--', label='demanded', linewidth=1)
        ax.set_ylabel(f'Drone {i+1}\nx (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('fig_x_positions.png', dpi=150)
    print('Saved fig_x_positions.png')

def plot_xy_trajectories():
    """Fig 2: x-y plane trajectories of all agents"""
    df = pd.read_csv('xy_trajectories.csv')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('X-Y plane trajectories (cf. Paper Fig 2)')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i in range(5):
        ax.plot(df[f'drone{i}_x'], df[f'drone{i}_y'],
                color=colors[i], label=f'Drone {i+1}', linewidth=1)
        ax.plot(df[f'drone{i}_x_desired'], df[f'drone{i}_y_desired'],
                '--', color=colors[i], alpha=0.5, linewidth=0.5)
        # Mark start position
        ax.plot(df[f'drone{i}_x'].iloc[0], df[f'drone{i}_y'].iloc[0],
                'o', color=colors[i], markersize=6)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('fig_xy_trajectories.png', dpi=150)
    print('Saved fig_xy_trajectories.png')

def plot_drone0_detail():
    """Detailed view of leader drone states"""
    df = pd.read_csv('drone0_states.csv')

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle('Leader (Drone 1) states: actual vs desired', fontsize=14)

    states = ['x', 'xdot', 'theta', 'y', 'ydot', 'phi', 'z', 'zdot', 'psi', 'r']
    labels = ['x (m)', 'ẋ (m/s)', 'θ (rad)', 'y (m)', 'ẏ (m/s)',
              'ϕ (rad)', 'z (m)', 'ż (m/s)', 'ψ (rad)', 'r (rad/s)']

    for idx, (state, label) in enumerate(zip(states, labels)):
        row = idx // 2
        col = idx % 2
        ax = axes[row][col]
        ax.plot(df['time'], df[state], label='actual', linewidth=1)
        ax.plot(df['time'], df[f'{state}_d'], '--', label='desired', linewidth=1)
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig('fig_drone0_states.png', dpi=150)
    print('Saved fig_drone0_states.png')

def plot_tracking_error():
    """Position tracking error over time for all drones"""
    df = pd.read_csv('x_positions.csv')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('X-position tracking error per drone')

    for i in range(5):
        error = df[f'drone{i}_x'] - df[f'drone{i}_x_desired']
        ax.plot(df['time'], error, label=f'Drone {i+1}', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('x error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('fig_tracking_error.png', dpi=150)
    print('Saved fig_tracking_error.png')

if __name__ == '__main__':
    # Check CSV files exist
    needed = ['x_positions.csv', 'xy_trajectories.csv', 'drone0_states.csv']
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print(f"Missing CSV files: {missing}")
        print("Run the simulation first: ./sim")
        sys.exit(1)

    plot_x_positions()
    plot_xy_trajectories()
    plot_drone0_detail()
    plot_tracking_error()
    print("\nAll plots saved.")
