import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set the aesthetic style of the plots
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Results data
dataset_sizes = [50000, 80000, 100000]
accuracy = [78.34, 78.89, 78.99]
f1_scores = [None, 78.89, 78.98]  # First F1 score is unknown
training_times = [None, 106.25, 120.03]  # First training time is unknown

# Create output directory if it doesn't exist
os.makedirs('enhanced_sentiment_results/visualizations', exist_ok=True)

# Set up the figure with a grid of subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Accuracy and F1 Score vs Dataset Size
ax1 = axes[0]
x = np.array(dataset_sizes) / 1000  # Convert to thousands for display

# Plot accuracy line
ax1.plot(x, accuracy, marker='o', linewidth=2, label='Accuracy')

# Plot F1 scores, skipping None values
valid_indices = [i for i, val in enumerate(f1_scores) if val is not None]
valid_sizes = [dataset_sizes[i]/1000 for i in valid_indices]
valid_f1 = [f1_scores[i] for i in valid_indices]
ax1.plot(valid_sizes, valid_f1, marker='s', linewidth=2, label='F1 Score')

ax1.set_xlabel('Dataset Size (thousands)')
ax1.set_ylabel('Score (%)')
ax1.set_title('Model Performance vs Dataset Size', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(size)}k' for size in x])
ax1.set_ylim(78, 79.5)  # Set reasonable y-axis limits
ax1.legend()
ax1.grid(True)

# Annotate the points with their values
for i, (size, acc) in enumerate(zip(x, accuracy)):
    ax1.annotate(f'{acc}%', 
                 xy=(size, acc), 
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9)

for size, f1 in zip(valid_sizes, valid_f1):
    ax1.annotate(f'{f1}%', 
                 xy=(size, f1), 
                 xytext=(0, -15),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9)

# Plot 2: Training Time vs Dataset Size
ax2 = axes[1]
valid_indices_time = [i for i, val in enumerate(training_times) if val is not None]
valid_sizes_time = [dataset_sizes[i]/1000 for i in valid_indices_time]
valid_times = [training_times[i] for i in valid_indices_time]

ax2.plot(valid_sizes_time, valid_times, marker='o', linewidth=2, color='green')
ax2.set_xlabel('Dataset Size (thousands)')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time vs Dataset Size', fontsize=14)
ax2.set_xticks([size for size in valid_sizes_time])
ax2.set_xticklabels([f'{int(size)}k' for size in valid_sizes_time])
ax2.grid(True)

# Annotate the points with their values
for size, time in zip(valid_sizes_time, valid_times):
    ax2.annotate(f'{time}s', 
                 xy=(size, time), 
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9)

# Add a trend line for training time
if len(valid_sizes_time) >= 2:
    z = np.polyfit(valid_sizes_time, valid_times, 1)
    p = np.poly1d(z)
    ax2.plot(valid_sizes_time, p(valid_sizes_time), linestyle='--', color='gray')
    
    # Calculate and display the slope
    minutes_per_10k = (p(valid_sizes_time[-1]) - p(valid_sizes_time[0])) / (valid_sizes_time[-1] - valid_sizes_time[0]) * 10
    ax2.text(0.05, 0.95, f'Time increase: ~{minutes_per_10k:.1f}s per 10k samples',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Add conclusion text
plt.figtext(0.5, 0.01, 
            "Conclusion: Increasing dataset size shows diminishing returns in model performance\n"
            "while training time continues to increase linearly.",
            ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('enhanced_sentiment_results/visualizations/performance_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved to enhanced_sentiment_results/visualizations/performance_vs_size.png") 