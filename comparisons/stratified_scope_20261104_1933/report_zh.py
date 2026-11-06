"""Render the complete Step 0 review tables and mathematical qualifications."""
from diagnostics_step0 import ROOT, read, markdown_table, write


def main():
    a,b,c=[read(ROOT/f'diagnostics/check_{key}.json') for key in 'abc']
    background=read(ROOT/'diagnostics/background_counts.json')
    historical=read(ROOT/'diagnostics/historical_real_scope.json')
    lines=['# ChoiceGrow Step 0：CPU 诊断报告','',
        '本报告只分析现有数据、训练日志和结构统计，不使用 GPU 训练或模型推理。现有主代码、数据和实验目录保持原样。Step 1 尚未启动，按用户要求等待本报告审阅。','',
        '记号：bn 为保留链中所有后继位置的频次背景；nn 为全链距离不超过 3 的逆距离邻居统计；no 为最后三个历史位置到最终目标的统计；nn_clean 只保留历史内部的距离不超过 3 的转移。h0 为 Root，hn 为原生 Infomap，ho 为旧随机三层，hs/hr 为本次谱分层／随机分层。','',
        '## A. 相同 epoch 的验证曲线与门控','',
        '以下 NDCG 均为验证集 NDCG@10。每个值取自该 run 的完整 choice_log.jsonl，并与 training.json 的最佳轮次核对；没有精确对应 epoch 时不插值。hn 份额顺序为社区／全目录，ho 为细层／粗层／全目录。','']
    for seed in (100,101,102):
        rows=[]
        for epoch in (100,200,280,300,400,500,600,800,1000,1200):
            row=[epoch]
            for method in ('h0','hn','ho'):
                v=next(x for x in a['matched_epochs'] if (x['kind'],x['seed'],x['epoch'])==(method,seed,epoch))
                row.append('未到达' if v['validation_ndcg10'] is None else f"{v['validation_ndcg10']:.6f}")
                if method!='h0':
                    row.append('—' if not v['shares'] else '/'.join(f'{100*s:.3f}%' for s in v['shares'].values()))
            rows.append(row)
        lines += [f'### seed {seed}','',markdown_table(['epoch','h0','hn','hn 份额','ho','ho 份额'],rows),'']
    lines += ['### 全日志最佳与实际停止轮次','',markdown_table(['方法','seed','最佳验证 NDCG@10','最佳 epoch','停止 epoch'],
        [[v['kind'],v['seed'],f"{v['best_validation_ndcg10']:.6f}",v['best_epoch'],v['stop_epoch']] for v in a['runs']]),'',
        markdown_table(['seed','共同验证轮数','hn 更高','Root 更高','最后共同 epoch','最后 hn−Root'],
        [[v['seed'],v['shared_validation_epochs'],v['hn_higher_epochs'],v['root_higher_epochs'],v['last_shared_epoch'],f"{v['last_shared_hn_minus_root']:+.6f}"] for v in a['paired_hn_vs_root']]),'',
        '**判断：hn 并非从头到尾低于 Root。** 共同 epoch 上 hn 经常更高，但各自早停后 Root 的最佳值更高。这提示训练时长与结构效果需要分开核验，不能据此证明“继续训练 hn 必然反超”。','',
        '### 对原假设的代码核验','',
        '1. `evaluate.py` 已在每次调用时通过 `fork_rng` 和由 seed/split/stream 派生的固定种子隔离、重置验证随机数；CPU 核验确认两次随机序列一致，退出后训练随机数状态恢复。CRN 复用随机数，不保证不同模型下的实际采样轨迹相同，也不使抽样误差归零。','',
        f"2. EMA 实际采用 `min(0.9999, (1+step)/(10+step))` 的渐增衰减率，直到 step 89,990 才达到上限。step 18,000 时 decay={a['protocol_audit']['ema_at_18000']['actual_decay']:.9f}，精确平均参数年龄约 {a['protocol_audit']['ema_at_18000']['exact_parameter_weight_mean_age_steps']:.1f} 步，即 Toys 约 29.5 个 epoch；不能从训练开始一律按 10,000 步／164 个 epoch 的滞后来解释。",'',
        '3. 18,000 步分别对应 Toys 296、Beauty 258、Sports 161 个 epoch。本次核验的九个 Toys h0/hn/ho run 都已超过该步数，因此新增此保护不会改变它们原有的停止时间。', '',
        f"4. 原报告 12 个不同方法／种子的 Toys run 中，最佳 epoch 与测试 NDCG 的 Spearman 相关为 {a['best_epoch_test_rank_association_original_12']['spearman_rho']:.4f}。这是混合方法与种子的探索性相关，不能当成早停的因果效应。",'',
        '## B. 目标自包含的量化','',
        '本节 Head/Tail 沿用既有长尾评测：训练保留历史＋目标频次低于严格阈值的物品属于 Tail。Toys 阈值为 12，长尾物品 9,343 个；本次验证集中长尾目标行是 989，不能与测试集的 986 行混用。排名是目标在原始 rho 分布中的全目录排名，不是去噪模型的推荐排名。','']
    rows=[]
    for v in b['statistics']:
        for group in ('head','tail'):
            g=v['groups'][group]
            rows.append([v['mode'],v['split'],group,g['rows'],f"{g['neighbor_target_mass']['mean']:.6f}",
                f"{g['rho_target_rank_gamma002']['mean']:.1f}",f"{g['rho_target_rank_gamma010']['mean']:.1f}",
                f"{100*g['same_native_community']['mean']:.2f}%",f"{100*g['same_clean_adjacent_community']['mean']:.2f}%"])
    lines += [markdown_table(['邻居','划分','目标组','行数','n_u(目标) 均值','rho 排名 γ=.02','rho 排名 γ=.10','目标与末历史同社区','clean 相邻边社区对照'],rows),'',
        'nn/no 的社区列使用原 hn 分区，nn_clean 使用历史内 lag≤3 构图的 Infomap。最后一列统一使用仅历史相邻边的 Infomap，用于区分“去掉最终目标边”与“将社区图扩为 lag≤3”这两个变化。各模式内部训练／验证／测试共用同一张表，但这不保证不同用户群的 n_u 密度或质量相同。','',
        '### 同一训练样本的配对扣边','',
        '仅从全训练计数中扣掉该样本自己指向最终目标的贡献，再重新加 self prior、选 Top-16 和归一化；该样本内部历史边及其他用户的贡献保留。这样可以直接测量目标自包含对邻居统计的影响。','']
    rows=[]
    for v in b['paired_own_final_edge_removal']:
        for group in ('head','tail'):
            g=v['groups'][group]
            rows.append([v['mode'],group,f"{g['target_mass_with_own_edges']['mean']:.6f}",
                f"{g['target_mass_without_own_edges']['mean']:.6f}",f"{g['own_edge_target_mass_increase']['mean']:.6f}",
                f"{g['rank_improvement_from_own_edges']['mean']:.1f}",f"{100*g['positive_own_edge_effect']['mean']:.2f}%"])
    lines += [markdown_table(['邻居','目标组','扣边前概率','扣边后概率','自身边增加概率','自身边改善排名','概率提高的样本占比'],rows),'',
        '**判断：训练目标自包含效应确实存在，长尾上幅度较大。** 这是训练结构中的自包含，不是使用验证／测试标签构图。Train/Val 差距还混有用户群与支持度差异；本报告未把它当作显著性检验或模型测试性能受损的因果证明。nn_clean 排除了直接最终监督标签边，但训练历史的自身影响和按要求保留的 bn 目标计数仍存在。社区层面没有逐用户重新运行 Infomap，因此其因果效应尚未单独识别。','',
        '## C. 分区平衡、保真度和验证块可预测性','',
        '构造统一使用 bn；社区图只使用训练历史内部 lag≤3 的逆距离边，亲和图 W+Wᵀ 去除自环。谱特征维度 32，频次层容量 M×K（M=20），10 次平衡分配迭代，seed=100。hs/hr 每个频次层的容量完全相同。完整频次层每社区恰好 20 个物品；末尾不满层采用共享的 floor/ceil 容量，避免参考片段中截断重复列导致余项偏向部分社区的问题。','',
        '本节保真度与 Head 占比 SD 的 Head 为按 bn 从高到低的前 floor(0.2N) 个物品；块预测概率的 Head/Tail 沿用既有长尾评测阈值。两套口径分别存入 JSON。保真度相对于同一协同谱空间的簇中心，不代表独立验证的商品语义。零谱向量物品的并列排名有任意性，JSON 另附去除这些物品后的结果。','']
    for dataset,v in c.items():
        lines += [f'### {dataset}','']
        rows=[]
        for method,g in v['partitions'].items():
            rows.append([method,g['clusters'],f"{g['block_size_min']}–{g['block_size_max']}",f"{g['block_mass_mean']:.6f}",
                f"{g['block_mass_std']:.6f}",f"{g['block_mass_min']:.6f}–{g['block_mass_max']:.6f}",f"{g['b_rank_head_fraction_std']:.6f}"])
        lines += [markdown_table(['分区','K','块大小范围','质量均值','质量 SD','质量范围','Head 占比 SD'],rows),'']
        rows=[]
        for method,g in v['partitions'].items():
            f=g['b_rank_fidelity'];p=g['validation_block_predictability']
            rows.append([method,*[f"{100*f[group][key]:.2f}%" for group,key in [('head','top1'),('head','top3'),('tail','top1'),('tail','top3')]],
                *[f"{p[group]['mass']['mean']:.6f}" for group in ('all','head','tail')]])
        lines += [markdown_table(['分区','Head Top1','Head Top3','Tail Top1','Tail Top3','验证目标块概率','Head 目标块概率','Tail 目标块概率'],rows),'',
            f"hs/hr 的验证目标块概率比 = **{v['hs_vs_hr_predictability_ratio']:.4f}**。对应块背景质量的最大差为 {v['paired_balance']['max_corresponding_block_mass_difference']:.6f}；历史图中零谱特征孤立物品 {v['construction']['graph']['isolated_items']} 个。",'']
    lines += ['### 历史 Real-Scope','',
        '三个数据集均找到了历史 Real-Scope。其 prepared 训练数组与当前 Grow 输入不同，尚未建立原始物品 ID 对应关系；以下只在各自历史背景与频次口径上统计，不跨映射计算当前用户的保真度或块预测概率。Toys 使用 real_100.pt，Beauty/Sports 使用现存最早的 real_200.pt。','']
    for dataset,v in historical.items():
        lines += [f'**{dataset}**','',markdown_table(['层','块数','大小范围','质量均值','质量 SD','质量范围','Head 占比 SD'],
            [[g['level'],g['clusters'],f"{g['size_min']}–{g['size_max']}",f"{g['block_mass_mean']:.6f}",f"{g['block_mass_std']:.6f}",
              f"{g['block_mass_min']:.6f}–{g['block_mass_max']:.6f}",f"{g['head_fraction_std']:.6f}"] for g in v['levels']]),'']
    lines += ['## D. 对背景 bn 的补充核对','',markdown_table(['数据集','bo 计数总量','bn 计数总量','bo 零计数物品','bn 零计数物品','bo=0 但历史频次≥5'],
        [[d,g['bo_total'],g['bn_total'],g['bo_zero_items'],g['bn_zero_items'],g['bo_zero_but_history_at_least5']] for d,g in background.items()]),'',
        'bn 显著增加可用计数、减少零计数物品；仅凭这个统计不能证明所有数据集上推荐或长尾 NDCG 必然提高。Toys 的 bn 平均计数实测为 7.984，而不是约 10。','',
        '## E. 必须区分的数学保证与经验假设','',
        '### 1. 频次分层保证容量，不保证质量完全相等','',
        '同一完整频次层有 MK 个物品，每社区分配 M 个，所以同层物品**数量**严格相同；层内 b 值通常不同，因此背景质量只近似均衡。若层内 b 的范围为 [a_s,z_s]，同层两个等容量社区的质量差至多 M(z_s−a_s)，全体完整层的上界为这些项之和。只有层内概率都相等等特殊情形，才能推出精确等质量。','',
        'hs/hr 保证相同的频段容量，不能称为“块内频率分布 100% 同构”或“只改变语义且其余统计完全一致”。实测残差已列出。三个数据集的 N 都不能被 K 整除，所以全目录所有块大小也不可能完全相等，只能相差最多 1。','',
        '### 2. 小社区不会仅因分母小而产生无界前向退出率','',
        r'令 r=(1−α)Λ，则 $-Q_{u,ii}=r\sum_\ell s_\ell\left(1-\rho_u(i)/\rho_u(C_\ell(i))\right)+\alpha(1-b_i)$。因 $0\le\rho_u(i)/\rho_u(C_\ell(i))\le1$，有 $0\le -Q_{u,ii}\le r+\alpha\approx2.256995$。所以“社区质量极小必然令前向退出率爆炸”不成立；小社区可能改变退出率、支持及多尺度间隔，需实测。','',
        r'对任意 $i\ne j$，$Q_{u,ij}\ge\alpha b_j>0$，因为 sqrt(count+1) 令 b 严格为正、reset_rate=0.035。社区因此不是严格封闭的吸收孤岛；这不保证有限时间内的长尾学习效果。','',
        '### 3. 固定份额是值得检验的干预，不是已经证明的改进','',
        '固定份额可使同一用户的 Q 在训练过程中保持不变，去掉动态门控与 EMA 去噪器之间由门控漂移引入的配对问题；但这不等于证明固定份额精度更高，也不意味着前向初始分布已处于平稳分布。固定份额的验证与 min_steps 的新增比较仍需后续实验。','',
        '## F. 数值与实现核验','',
        '6 项 CPU 测试覆盖历史内／全链／末目标计数、逐样本配对扣边、带并列的稀疏排名、尾部容量一致、普通平衡 LP 的整数容量解。历史邻居表用原始 float32 代码重放，与保存的 IDs 和权重逐位一致；float64 诊断表的支持集相同，最大归一化差约 1.02×10⁻⁷。','',
        '扣边概率使用重新累加非负组成行，避免“归一化概率相减”留下约 10⁻¹⁷ 的伪非零值。回归样例曾因此把并列排名 17 错算为 7；修正后目标概率精确为 0，排名为 17。','',
        '## G. 当前交付与下一步','',
        'Step 0 已完成。代码、JSON/Markdown 报告和三数据集 NPZ 分区都在本 comparison 目录。按用户明确要求，等待检查 Step 0 后再启动 Step 1 的两个 Toys seed100 Root 校准 run；Step 2 仍须以 Step 1 的验证 NDCG@10 选择 γ，不能用测试集选参。固定份额训练、min_steps 及 Step 1/2 runner 此时尚未接入或执行，不将其误报为完成。','']
    (ROOT/'diagnostics/report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'diagnostics/report_zh.md')


if __name__=='__main__':
    main()
