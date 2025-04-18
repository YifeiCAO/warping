import random
from data import GridDataGenerator

class MetaRLTaskSampler:
    def __init__(self, args, k_support=8, k_query=16):
        """
        Meta-RL Task Sampler: 提供每个任务的支持集（rank=1）和查询集（rank>1）
        
        Args:
            args: argparse 结构体，包含 grid_size, use_images 等配置
            k_support: 每个任务中支持集的样本数（rank=1）
            k_query: 每个任务中查询集的样本数（rank>1）
        """
        self.k_support = k_support
        self.k_query = k_query

        # 构建原始样本池（包含两个任务：ctx=0 和 ctx=1）
        self.generator = GridDataGenerator(
            training_regime='train_all',  # 用全部样本构建任务池
            size=args.grid_size,
            use_images=args.use_images,
            image_dir=args.image_dir,
            inner_4x4=args.inner_4x4
        )

        all_samples = self.generator.train

        # 按 context（任务）分组
        self.tasks = {0: [], 1: []}  # 0: 按 x 排序，1: 按 y 排序
        for sample in all_samples:
            ctx, f1, f2, y, info = sample
            self.tasks[ctx].append(sample)

        # 打乱每个任务内样本顺序
        for ctx in self.tasks:
            random.shuffle(self.tasks[ctx])

    def sample_task(self, task_id):
        """
        采样一个指定任务的 support + query set
        """
        samples = self.tasks[task_id]
        rank1_samples = [s for s in samples if abs(s[1][task_id] - s[2][task_id]) == 1]
        rankn_samples = [s for s in samples if abs(s[1][task_id] - s[2][task_id]) > 1]

        support_set = random.sample(rank1_samples, min(self.k_support, len(rank1_samples)))
        query_set = random.sample(rankn_samples, min(self.k_query, len(rankn_samples)))

        return support_set, query_set

    def sample_batch(self, batch_size):
        """
        一次采多个任务（混合 ctx=0 和 ctx=1），用于 batch meta-training
        返回：[(support_set, query_set), ...] 长度为 batch_size
        """
        task_batch = []
        for _ in range(batch_size):
            task_id = random.choice([0, 1])
            support, query = self.sample_task(task_id)
            task_batch.append((task_id, support, query))
        return task_batch
