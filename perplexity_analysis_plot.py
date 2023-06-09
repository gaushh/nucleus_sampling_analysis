import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(csv_file, output_folder):
    x = []
    y = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(float(row['ppl']))
            print("row['repetitiveness_score'] :::", row['repetitiveness_score'])
            y.append(float(row['repetitiveness_score']))

    plt.scatter(x, y)
    plt.xlabel('ppl')
    plt.ylabel('repetitiveness_score')
    plt.title('Scatter Plot: repetitiveness_score vs ppl')

    # Set x-axis to log scale
    plt.xscale('log')

    # Calculate coefficients of best fit curve
    coeffs = np.polyfit(np.log10(x), y, 1)

    # Generate x values for the curve
    curve_x = np.logspace(np.log10(min(x)), np.log10(max(x)), num=100)

    # Evaluate the polynomial at the x values
    curve_y = np.polyval(coeffs, np.log10(curve_x))

    # Plot the curve of best fit
    plt.plot(curve_x, curve_y, color='red', label='Best Fit')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the scatter plot in the output folder
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file))[0] + '.png')
    plt.savefig(output_file)
    plt.close()

def main():
    input_folder = 'results/perplexity_analysis'
    output_folder = 'results/perplexity_analysis/scatter_plots_log'

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                plot_scatter(csv_file, output_folder)

if __name__ == '__main__':
    main()
