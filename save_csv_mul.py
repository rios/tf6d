import pandas as pd

import pandas as pd


def process_data(file_path, output_path, K=1,  metric_type='conf'):
    """
    处理数据以生成每个 instance_id 的 shared_score，然后基于 shared_score
    保留每个 (scene_id, im_id, obj_id) 组合中 shared_score 最大的 instance_id 的所有行。
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 第一步：按照 instance_id 分组，计算前 K 大 score 的和作为 shared_score
    data['shared_score'] = data.groupby('instance_id')[metric_type] \
        .transform(lambda x: x.nlargest(K).sum())

    # 第二步：在每个 (scene_id, im_id, obj_id) 组合中，保留具有最大 shared_score 的 instance_id 的所有行
    # 找到每个 (scene_id, im_id, obj_id) 组合的最大 shared_score
    data['max_shared_score'] = data.groupby(['scene_id', 'im_id', 'obj_id'])['shared_score'].transform('max')

    # 过滤出 shared_score 等于 max_shared_score 的所有行
    filtered_data = data[data['shared_score'] == data['max_shared_score']]

    # 删除不需要的列
    columns_to_drop = ['pnp_inliners', 'GT_error', 'conf', 'shared_score', 'max_shared_score', 'instance_id']
    filtered_data = filtered_data.drop(columns=[col for col in columns_to_drop if col in filtered_data.columns])

    # 将结果保存至文件
    filtered_data.to_csv(output_path, index=False)
    print(f"处理后的数据已保存到 {output_path}")


def process_data_top_k(file_path, output_path, m=1, k=1, metric_type='pnp_inliners'):
    """
    Processes data to retain the top k highest scoring rows for each instance_id,
    using the sum of the top-m values of either 'score' or 'pnp_inliners' as the metric.
    """
    data = pd.read_csv(file_path)

    # Step 1: Compute the sum of top-m values of the chosen metric ('score' or 'pnp_inliners') for each 'instance_id' group
    grouped = data.sort_values(by=metric_type, ascending=False) \
        .groupby(['scene_id', 'im_id', 'obj_id', 'instance_id']) \
        .head(m).groupby(['scene_id', 'im_id', 'obj_id', 'instance_id']) \
        .agg(metric=(metric_type, 'sum')).reset_index()

    # Step 2: Keep the highest metric for each unique 'scene_id', 'im_id', 'obj_id'
    highest_metric = grouped.loc[grouped.groupby(['scene_id', 'im_id', 'obj_id'])['metric'].idxmax()]

    # Step 3: Merge with original data, retain top k rows for each 'instance_id'
    filtered = data.merge(highest_metric[['scene_id', 'im_id', 'obj_id', 'instance_id']],
                          on=['scene_id', 'im_id', 'obj_id', 'instance_id'])
    top_k_data = filtered.sort_values(by=['instance_id', 'score'], ascending=[True, False]) \
        .groupby('instance_id').head(k).reset_index(drop=True)
    top_k_data.drop(columns=['pnp_inliners',
                             'GT_error',
                             'conf']).to_csv(output_path, index=False)

    print(f"Top {k} rows per instance_id saved to {output_path}")


# Usage Examples
input_file = './logs/results_fastsam_final/AllMASt3R8Conf-lmo-test_bop.csv'
out_top1_file = './logs/results_fastsam_final/Top1MASt3R8Conf_lmo-test_bop.csv'
# out_top1_file = '/data/weijian/Codes/Pose/gigapose/gigaPose_datasets/results/large_lmodinov2/predictions/Top1MASt3R8Conf_lmo-test_bop.csv'

out_topk_file = '/data/weijian/Codes/Pose/gigapose/gigaPose_datasets/results/large_lmodinov2/predictions/TopkMul-lmo-test_bopMultiHypothesis.csv'

# Using the sum of top-m 'pnp_inliners' as the metric
# process_data(input_file, out_top1_file, m=1, metric_type='pnp_inliners')
# process_data_top_k(input_file, out_topk_file, m=3, k=3, metric_type='pnp_inliners')

# Using the sum of top-m 'score' as the metric
process_data(input_file, out_top1_file, K=3, metric_type='score')
process_data_top_k(input_file, out_topk_file, m=3, k=6, metric_type='score')
