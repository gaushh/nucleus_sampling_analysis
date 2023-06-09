import os
import matplotlib.pyplot as plt
import pandas as pd

output_csv = 'quantitative_results.csv'
qual_df = pd.read_csv(output_csv)

# Define a function to split the filename and handle errors
# print(qual_df)
def split_filename(filename, index):
    parts = filename.split('_')
    if len(parts) > index:
        return parts[index]
    else:
        return None

# Split the 'filename' column into multiple columns
qual_df['characteristic_1'] = qual_df['File'].apply(lambda x: split_filename(x, 0))  # Splitting by underscore and taking the first part
qual_df['characteristic_2'] = qual_df['File'].apply(lambda x: split_filename(x, 1))  # Splitting by underscore and taking the second part
qual_df['characteristic_3'] = qual_df['File'].apply(lambda x: split_filename(x, 2))  # Splitting by underscore and taking the third part
qual_df['characteristic_4'] = qual_df['File'].apply(lambda x: split_filename(x, 3))


def split_val(x):
    try:
        # print(x)
        splitted = x.split("=")[1]
        return float(splitted)
    except:
        return None


qual_df['characteristic_6'] = qual_df['characteristic_3'].apply(lambda x: split_val(x))

# print("characteristic_6 :: ", qual_df['characteristic_6'])
# Split 'characteristic_4' to extract boolean value

def split_x(x):
    if x:
        splitted = x.split("=")[0]
        return splitted[0] != "t"
    else:
        return None

qual_df['characteristic_5'] = qual_df['characteristic_4'].apply(lambda x: split_x(x))

# Create a 'results' folder if it doesn't exist
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# EDA for top-p
qual_df_modified = qual_df[(qual_df["characteristic_2"] == "topp") & (qual_df["characteristic_5"] == True)]

# Group the DataFrame by 'n' values
grouped_df = qual_df_modified.groupby('n')

# Create subfolders for top-p plots
topp_folder = os.path.join(results_folder, 'topp')
os.makedirs(topp_folder, exist_ok=True)

# Iterate over each group and plot line plots
for n_value, group in grouped_df:
    group_sorted = group.sort_values('characteristic_6')  # Sort the group by 'characteristic_3'
    # print(group_sorted)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(group_sorted['characteristic_6'], group_sorted['Repetition Percentage'])
    ax.set_title(f'n={n_value}')
    ax.set_xlabel('p_val')
    ax.set_ylabel('Repetition Percentage')
    plt.tight_layout()

    # Save the plot in the 'topp' subfolder
    filename = f'topp_n={n_value}.png'
    filepath = os.path.join(topp_folder, filename)
    plt.savefig(filepath)
    plt.close()

# EDA for top-k without temperature
qual_df_modified_k = qual_df[(qual_df["characteristic_2"] == "topk") & (qual_df["characteristic_5"] == True)]

# Extract 'k_val' from 'characteristic_3'
# qual_df_modified_k['k_val'] = qual_df_modified_k['characteristic_6'].apply(lambda x: int(x.split("=")[1]))

# Group the DataFrame by 'n' values
grouped_df_k = qual_df_modified_k.groupby('n')

# Create subfolders for top-k without temperature plots
topk_no_temp_folder = os.path.join(results_folder, 'topk_no_temp')
os.makedirs(topk_no_temp_folder, exist_ok=True)

# Iterate over each group and plot line plots
for n_value, group in grouped_df_k:
    group_sorted = group.sort_values('characteristic_6')  # Sort the group by 'k_val'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(group_sorted['characteristic_6'], group_sorted['Repetition Percentage'])
    ax.set_title(f'n={n_value}')
    ax.set_xlabel('characteristic_6')
    ax.set_ylabel('Repetition Percentage')
    plt.tight_layout()

    # Save the plot in the 'topk_no_temp' subfolder
    filename = f'topk_no_temp_n={n_value}.png'
    filepath = os.path.join(topk_no_temp_folder, filename)
    plt.savefig(filepath)
    plt.close()

# EDA for top-k with temperature
qual_df_modified_k = qual_df[(qual_df["characteristic_2"] == "topk") & (qual_df["characteristic_5"] == False)]

# Extract 'k_val' from 'characteristic_3'
# qual_df_modified_k['k_val'] = qual_df_modified_k['characteristic_6'].apply(lambda x: int(x.split("=")[1]))

# Group the DataFrame by 'n' values
grouped_df_k = qual_df_modified_k.groupby('n')

# Create subfolders for top-k with temperature plots
topk_with_temp_folder = os.path.join(results_folder, 'topk_with_temp')
os.makedirs(topk_with_temp_folder, exist_ok=True)

# Iterate over each group and plot line plots
for n_value, group in grouped_df_k:
    group_sorted = group.sort_values('characteristic_6')  # Sort the group by 'k_val'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(group_sorted['characteristic_6'], group_sorted['Repetition Percentage'])
    ax.set_title(f'n={n_value}')
    ax.set_xlabel('k_val')
    ax.set_ylabel('Repetition Percentage')
    plt.tight_layout()

    # Save the plot in the 'topk_with_temp' subfolder
    filename = f'topk_with_temp_n={n_value}.png'
    filepath = os.path.join(topk_with_temp_folder, filename)
    plt.savefig(filepath)
    plt.close()
