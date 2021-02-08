# -*- coding: utf-8 -*-

"""框架在启动时会加载的基础配置

其中，包含了框架支持的配置项及其默认值
"""

BASE_CONFIG = {
    # Trial配置
    # 并行训练数量
    'parallel_num': 5,
    # 初始探索的实验数量，在不同算法中具有不同的具体含义，但调优过程通常会由 baselines 实验开始，
    # 然后创建 init_trial_number 个根据 tuning_params 配置的初始探索实验
    # NOTE: `init_trial_number` 是否允许设置为 0 由具体 scheduler 决定
    'init_trial_number': 10,
    # 最大搜索实验数量，使用算法策略会不断生成新的搜索实验，当创建的trial数量达到上限后就会
    # 停止整个study过程
    'max_trial_number': 20,
    'target_scores': 0.9,
    # 初始 checkpoint 配置，支持使用单个或多个checkpoint:
    #   str: 传递单个checkpoint
    #   list: 传递多个checkpoint
    'init_checkpoint': '',

    # 算法配置
    # 超参数调优算法配置:
    #   RandomSearch / GridSearch: bf
    #   BayesOptimize: bo
    #   PBT: pbt
    #   EarlyStop: multhreshold_es
    #   Customized: other
    'scheduler_alg': 'bf',
    # es_config
    # 进入提前停止判断逻辑的条件设置dict，若不设置条件，表示每次调度都会判断
    # 其中每个key-value都表示一个条件，多个条件之间是"与"关系
    # - 支持的条件类型（key）有step、duration；
    # - 支持的条件取值（value）有int、list：
    #   - int表示每隔指定步数/时间就进行一次判断；
    #   - list表示进行判断的各个取值节点，需要按照严格的升序进行排序。
    #   如:[60, 120, 150, 300]表示分别在trial执行时间到60、120、150、300分钟时进行es判断
    'es_check_threshold': {
        # 步数限制条件（步）
        # 'step': 0,
        # 时间限制条件（分钟）
        # 'duration': 0,
    },
    # 在ES算法配置中，使用的时间单位，支持s/m/h/d，默认为m（分钟）
    'es_duration_unit': 'm',
    # multhreshold_es算法中，每次执行es判断时的ranking得分阈值
    # NOTE：
    # - ranking得分指的是根据ranking配置，实时的所有metrics*权重之和，所得到的score
    # - 这里是一个阈值list，需要配合es_check_threshold进行配置。例：
    #   es_check_threshold.duration: [30, 60, 150, 300]，
    #   es_mul_thresholds: [75.5, 110, 130, 180]
    #   将会在训练耗时30m时检查score是否达到75.5，60m时检查score是否达到110...
    #   （若两者长度不相同，以短的为准）
    # - 这里的list元素需要按照ranking_method配置进行严格排序：若max，则升序；若min，则降序
    'es_mul_thresholds': [],
    # pbt_config
    # 每次执行 update (exploit or explore) 操作时，参与操作的个体比例（实际运算时会
    # 向上取整），取值范围为 (0, 0.5)
    'pbt_mix_range': 0.3,
    # 每个 round 之间间隔的步数
    'pbt_round_step': 10000,
    # 是否在个体更新的评估环节使用fitness sharing优化多样性
    'pbt_is_enable_fitness_sharing': False,
    # fitness sharing优化中计算niche count时每个sample与其余所有sample的相似距离阈值
    'pbt_niche_sigma': 0.1,
    # fitness sharing优化中计算niche count时控制sharing function形态的超参
    'pbt_niche_alpha': 1.0,
    # bo_config
    # 每次重新拟合后，会通过随机采样 bo_sample_size 次数来获取激活函数得分为 top K 的采样
    'bo_sample_size': 1000,
    # 激活函数类型，支持 `['ucb', 'ei', 'poi']`
    'bo_acq_func_type': 'ucb',
    # 每次重新拟合后，产生新的实验的数量上限（由于在 get_tok_k 操作中会排除相似程度小于阈值的
    # 采样，导致了返回的采样值可能小于 bo_new_trial_parallel_num，但至少会为 1）
    'bo_new_trial_parallel_num': 1,
    # 每次重新拟合后，在进行随机采样获取激活函数得分 top k 的同时，会进行 bo_optimize_times
    # 指定的次数的最优化探索
    'bo_optimize_times': 10,
    # 每次重新拟合后，在 get_tok_k 操作中会排除相似程度小于 bo_sample_similar_threshold
    # 的采样
    'bo_sample_similar_threshold': 0.01,

    # 先验数据配置（默认为空list）
    # e.g.
    # 'scheduler_priors': [
    #     {
    #         # params必须要包含search_config中的tuning_params
    #         "params": {"vec_eta": 0.04129, "alpha": 0.00121},
    #         # 这里的result相当于ranking_metrics中计算的最终得分
    #         "result": 0.693725
    #     },
    #     {
    #         "params": {"vec_eta": 0.02995, "alpha": 0.00165},
    #         "result": 0.693122
    #     },
    # ],
    # 'scheduler_priors': 'prior.json',
    'scheduler_priors': [],

    # 将结果保存为priors时，指定输出文件，默认输出到当前目录下的priors.json文件中
    'result_as_priors_file': '',
    # 是否自动保存训练结果为priors格式，默认禁用
    'is_save_result_as_priors_enable': False,

    # 训练结果评价方法配置
    # 指定结果名称的多级排序: default
    'ranking.ranking_class': 'default',
    # 评价指标：
    #   dict: 用于多级加权评价，key为指标名称，value为权重（0～1）
    #   str:  用于单指标评价，value为指标名称
    'ranking.ranking_metrics': {},
    # 评价方法：
    # - max表示metrics得分越大表示能力越强（默认值）
    # - min表示metrics得分越小表示能力越强
    'ranking.ranking_method': 'max',

    # -------------------------------search_config------------------------------
    # 当搜索类型为连续搜索时，该参数指定每次扰动的幅度，即
    # `new_value = old_value +/- perturb_factor *（end - start）`
    'search.perturb_factor': 0.2,
    # 当搜索类型为离散搜索时，该参数指定每次扰动的索引长度，即
    # `new_idx = (old_idx + len(choices) + perturb_step) % len(choices)`
    'search.perturb_step': 1,
    # 每次执行 perturb 时，该参数指定直接重新随机采样的概率
    'search.resample_prob': 0.1,
    # 浮点数参数要保留的小数位数，默认保留5位小数
    'search.float_decimal': 5,
    # 该参数指定自定义表达式的参数的前缀标识符，所有表达式类型的参数在定义时要加上该参数值作为前缀
    'search.expression_flag': '$CUSTOMIZED$',

    # baselines 是一个 list，其中每一项都表示一组直接指定的尝试训练的超参数配置。
    # - 在每一组配置中，每个参数的名称必须只包含数字、字母、`_` 或 `#`
    # - baselines 配置支持使用**自定义表达式**进行定义。在此类配置中，参数值的定义是一个
    #   string，且需要以 `expression_flag` 配置的标识符开头（否则无法识别），在表达式中可以
    #   使用 `${}` 来引用一个已经在 baseline 中定义了的参数变量（自定义表达式引用的变量必须是
    #   值确定的参数）。在每一个 trial 中，自定义表达式类型的参数都会将其中引用的各个变量自动解
    #   析为当前 trial 的取值。
    # - 由于所有 trial 的目的都是为了得到 tuning_param 中配置的调优参数的最优采样，因此，
    #   **在 baselines 中的每一组配置要求都必须要包含下边 tuning_params 中配置的调优参数**。
    # - baselines 配置允许为空，当其设置为空 list 时，搜索实验的参数将只包含 tuning_params
    #   中包含的参数。
    # e.g.
    # 'search.baselines': [
    #     {
    #         'GAMMA': 0.996,
    #         'LAMDA': 0.996,
    #         'vec_eta': 0.02,
    #         'alpha': 0.02,
    #         'doc_tower_slots#vec_lambda2': 0.0,
    #         'TEST': '$CUSTOMIZED$${doc_tower_slots#vec_lambda2}'
    #     },
    #     {
    #         'BATCH_SIZE': 64,
    #         'INIT_CLIP_PARAM': 0.0,
    #         'GAMMA': 0.996,
    #         'LAMDA': 0.03,
    #         'vec_eta': 0.02,
    #         'alpha': 0.02,
    #         'doc_tower_slots#vec_lambda2': 0.1,
    #         'TEST': '$CUSTOMIZED$${doc_tower_slots#vec_lambda2}+${LAMDA}'
    #     },
    # ],
    'search.baselines': [],

    # tuning_params 是一个 list，其中每一项都表示**一个**要调优的超参数配置字典对象，
    # 在进行参数搜索时，在第一组 baseline 的基础上进行的参数值更新就是由这部分配置来决定的。
    # e.g.
    # 'search.tuning_params': [
    #       {
    #           'name': 'LAMDA',
    #           'type': 'continuous',
    #           'start': 0.95,
    #           'end': 1.0,
    #           'distribution': 'uniform'
    #       },
    #       {
    #           'name': 'GAMMA',
    #           'type': 'discrete',
    #           'choices': [0.91, 0.92, 0.93, 0.94, 0.95, 0.96],
    #           'method': 'force',
    #       },
    # ],
    'search.tuning_params': [],

}
