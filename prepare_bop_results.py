import pandas as pd


def process_csv(input_file_path, output_file_path_n, output_file_path_k, n=3, k=2):
    # Load the input CSV file
    df = pd.read_csv(input_file_path)

    # Remove 'pnp_inliers' column if it exists
    if 'pnp_inliners' in df.columns:
        df = df.drop(columns=['pnp_inliners'])

    # Create the instance_id column based on scene_id, im_id, and obj_id
    df['instance_id'] = df.groupby(['scene_id', 'im_id', 'obj_id']).ngroup()

    # Function to select the row with the highest score for every n rows within the same instance_id
    def select_highest_score_per_group(group):
        return group.groupby(group.index // n).apply(lambda x: x.loc[x['score'].idxmax()])

    # Apply this function to each instance_id group
    selected_rows_n = df.groupby('instance_id', group_keys=False).apply(select_highest_score_per_group)

    # Save the n-based selection to the first output CSV
    selected_rows_n.to_csv(output_file_path_n, index=False)

    # Now select the top k rows from the selected n rows based on the highest score
    # selected_rows_k = selected_rows_n.groupby('instance_id', group_keys=False).apply(
    #     lambda x: x.nlargest(k, 'score'))

    selected_rows_k = selected_rows_n.groupby('instance_id', group_keys=False).apply(
        lambda x: x.nsmallest(k, 'GT_error'))

    # Save the k-based selection to the second output CSV
    selected_rows_k.to_csv(output_file_path_k, index=False)


# Function to process the CSV file and select the top k rows based on score for each instance_id
def process_top_k_csv(input_file_path, output_file_path, k=2, instance_id=False):
    # Load the input CSV file
    df = pd.read_csv(input_file_path)

    # Remove '.tif' extension from 'im_id' if present
    df['im_id'] = df['im_id'].astype(str).apply(lambda x: x.replace('.tif', '') if '.tif' in x else x)

    # Calculate the average time for each "scene_id, im_id" group and divide by 2
    df['time'] = df.groupby(['scene_id', 'im_id'])['time'].transform('mean')

    # # Select the top k rows based on the score within each 'instance_id' group
    selected_rows_k = df.groupby(
        'instance_id', group_keys=False
    ).apply(lambda x: x.nlargest(k, 'score'))

    # Calculate the new metric and store it in a temporary column
    # df['combined_score'] = df['score']
    # # Group by 'instance_id' and select the top `k` rows based on 'combined_score'
    # selected_rows_k = df.groupby(
    #     'instance_id', group_keys=False
    # ).apply(lambda x: x.nlargest(k, 'combined_score'))
    # #
    # selected_rows_k = df.groupby(
    #     'instance_id', group_keys=False
    # ).apply(lambda x: x.nsmallest(k, 'GT_error'))

    # Define columns to drop based on `instance_id` flag
    columns_to_drop = ['pnp_inliners', 'GT_error', 'combined_score', 'conf'] if instance_id else ['pnp_inliners',
                                                                                                  'GT_error',
                                                                                                  'instance_id',
                                                                                                  'combined_score',
                                                                                                  'conf']
    selected_rows_k = selected_rows_k.drop(columns=[col for col in columns_to_drop if col in selected_rows_k.columns])

    # Save the selected rows to the output CSV
    selected_rows_k.to_csv(output_file_path, index=False)
    print(f"Processed file saved to {output_file_path}")


# Example usage:
dataset_name = 'tless'
file_name = 'MASt3R401-%s-test_bop' % (dataset_name)
input_file_path = './logs/results_fastsam_final/All%s.csv' % (file_name)
output_file_path_n = './logs/results_fastsam_final/Group%s.csv' % (file_name)
# output_file_path_n = './results/large_%sdinov2/predictions/Group%s.csv' % (dataset_name, file_name)

output_file_path_k = './results/large_%spoe3r/predictions/TopK%sMultiHypothesis.csv' % (
dataset_name,
file_name)

# Set the number of top rows to select (k)
process_top_k_csv(input_file_path, output_file_path_n, k=1)
# process_top_k_csv(input_file_path, output_file_path_k, k=5, instance_id=True)

# Set the number of rows to group together (n) and the number of top rows to select (k)
# n = 8  # Group size
# k = 10 # Top selection size
# process_csv(input_file_path, output_file_path_n, output_file_path_k, n, k)


