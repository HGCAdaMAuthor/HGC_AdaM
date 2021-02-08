# -*- coding: utf-8 -*-

"""框架在启动时会加载的基础配置

其中，包含了框架支持的配置项及其默认值
"""

BASE_CONFIG = {
    # Trainer配置
    # 并行训练数量
    'parallel_num': 4,

    # 初始探索的实验数量，在不同算法中具有不同的具体含义，但调优过程通常会由 baselines 实验开始，
    # 然后创建 init_trial_number 个根据 tuning_params 配置的初始探索实验
    # NOTE: `init_trial_number` 是否允许设置为 0 由具体 scheduler 决定
    'init_trial_number': 4,
    # 初始 checkpoint 配置，支持使用单个或多个checkpoint:
    #   str: 传递单个checkpoint
    #   list: 传递多个checkpoint
    'target_scores': 0.013,

    # 算法配置
    # 超参数调优算法配置:
    #   RandomSearch / GridSearch: bf
    #   BayesOptimize: bo
    #   PBT: pbt
    #   EarlyStop: multhreshold_es

    #   Customized: other
    'scheduler_alg': 'pbt',
    # pbt_config
    # 每次执行 update (exploit or explore) 操作时，参与操作的个体比例（实际运算时会
    # 向上取整），取值范围为 (0, 0.5)
    'pbt_mix_range': 0.3,
    # 每个 round 之间间隔的步数
    'pbt_round_step': 1000,
    # 是否在个体更新的评估环节使用fitness sharing优化多样性
    'pbt_is_enable_fitness_sharing': False,
    # fitness sharing优化中计算niche count时每个sample与其余所有sample的相似距离阈值
    'pbt_niche_sigma': 0.1,
    # fitness sharing优化中计算niche count时控制sharing function形态的超参
    'pbt_niche_alpha': 1.0,

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
    'ranking.ranking_metrics': "loss",
    # 评价方法：
    # - max表示metrics得分越大表示能力越强（默认值）
    # - min表示metrics得分越小表示能力越强
    'ranking.ranking_method': 'min',

    # -------------------------------search_config------------------------------
    # 当搜索类型为连续搜索时，该参数指定每次扰动的幅度，即
    # `new_value = old_value +/- perturb_factor *（end - start）`
    'search.perturb_factor': 0.0001,
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
    'search.baselines': [
		{
			'lr': 0.0055,
			'hidden1': 64,
			'hidden2': 64
		}
	],

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
    'search.tuning_params': [
        {
            'name': 'lr',
            'type': 'continuous',
            'start': 0.0001,
            'end': 0.01,
            'distribution': 'uniform'
        },
        {
            'name': 'hidden1',
            'type': 'discrete',
            'choices': [64, 128, 256, 512],
            'method': 'force',
        },
        {
            'name': 'hidden2',
            'type': 'discrete',
            'choices': [16, 32, 64, 128],
            'method': 'force',
        },
    ],

}
