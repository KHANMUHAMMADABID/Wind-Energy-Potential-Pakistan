import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set global font to Times New Roman and font sizes to 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 7  # smaller for many stations

# Load the wind speed data from the CSV file
file_path = '1-Wind-Speed-at-different-height.csv'
data = pd.read_csv(file_path)

# Set the station names as the index for easier plotting
data = data.set_index(data.columns[0])

# Clean column names to just the integer hub heights (e.g., 10, 20, ..., 100)
data.columns = [col.split(' ')[-3].replace('m', '') for col in data.columns]

# Convert all data to numeric (in case of any non-numeric values)
data = data.apply(pd.to_numeric, errors='coerce')

# Prepare X-tick labels: only numbers (no 'm')
xtick_labels = [int(col) for col in data.columns]

# Set figure size: width small, height enough for all stations
plt.figure(figsize=(5, min(0.35 * len(data.index), 8)))  # cap height at 8 inches for journals

# Create the heatmap
ax = sns.heatmap(
    data,
    annot=False,           # No values in cells
    cmap="YlGnBu",         # Color map for the heatmap
    linewidths=0.5,        # Line width between cells
    cbar_kws={'label': 'Wind Speed (m/s)', 'shrink': 0.7},
    square=False
)

# Set custom X-tick labels (numbers only, font size 10)
ax.set_xticklabels(xtick_labels, fontsize=10, fontname='Times New Roman')
# Set custom Y-tick labels (all station names, font size 7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, fontname='Times New Roman')

# Set axis labels and title (all font size 10)
ax.set_xlabel('Hub Height (m)', fontsize=10, fontname='Times New Roman')
ax.set_ylabel('Station Name', fontsize=10, fontname='Times New Roman')
ax.set_title('Wind Speed at Different Hub Heights', fontsize=10, fontname='Times New Roman', pad=10)

# Add a black frame around the heatmap using the axes patch
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(0.7)

# Set all spines for a more pronounced frame
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.7)

# Add a black border to the colorbar (legend)
cbar = ax.collections[0].colorbar
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(0.7)

plt.tight_layout()

plt.savefig('wind_speed_heatmap.png', dpi=600, bbox_inches='tight')
plt.show()
