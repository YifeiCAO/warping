import torch
import torch.nn as nn
import numpy as np
from data_meta import MetaRLTaskSampler
from utils import log
from test import test
from analyze import analyze


def train_meta_rl(run_i, model, args):
    # 初始化数据采样器
    sampler = MetaRLTaskSampler(args, k_support=args.k_support, k_query=args.k_query)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().to(args.device)

    train_losses, train_accs, test_accs, analyze_accs, analyses = [], [], [], [], []
    step_i = 0

    while step_i < args.n_steps:
        task_batch = sampler.sample_batch(batch_size=args.meta_batch_size)

        optimizer.zero_grad()
        batch_loss = 0

        for task_id, support_set, query_set in task_batch:
            model.reset_hidden(batch_size=1)  # 重置 LSTM 状态

            # Step 1: Feed support set（不计算 loss）
            for ctx, f1, f2, y, info in support_set:
                ctx = torch.tensor(ctx).to(args.device).unsqueeze(0).long()
                f1 = f1.to(args.device)
                f2 = f2.to(args.device)
                y_prev = torch.tensor([0]).to(args.device)  # 假设第一个 y_prev 是 0
                _ = model(ctx, f1, f2, y_prev)  # 只更新内部状态，不计算 loss

            # Step 2: Feed query set + accumulate loss
            task_loss = 0
            for ctx, f1, f2, y, info in query_set:
                ctx = ctx.to(args.device)
                f1 = f1.to(args.device)
                f2 = f2.to(args.device)
                y = y.to(args.device)
                y_prev = y  # 简单设置为 ground-truth，可换成 previous prediction

                y_hat, _ = model(ctx, f1, f2, y_prev)
                task_loss += loss_fn(y_hat, y)

            batch_loss += task_loss / len(query_set)

        # Update model
        batch_loss /= args.meta_batch_size
        batch_loss.backward()
        optimizer.step()

        # Logging
        if step_i % args.print_every == 0:
            print(f"[Meta-RL] Run: {run_i}, Step: {step_i}, Loss: {batch_loss.item():.4f}")
            train_losses.append(batch_loss.item())

        if step_i % args.test_every == 0 or step_i == args.n_steps - 1:
            train_acc = test(model, sampler.generator.train, args)
            test_acc = test(model, sampler.generator.test, args)
            analyze_acc = test(model, sampler.generator.analyze, args)
            log(train_acc['acc'], test_acc['acc'], analyze_acc['acc'])
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            analyze_accs.append(analyze_acc)

        if step_i % args.analyze_every == 0 or step_i == args.n_steps - 1:
            analysis = analyze(model, sampler.generator.analyze, args, final_step=True)
            analyses.append(analysis)

        step_i += 1

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'analyze_accs': analyze_accs
    }

    return results, analyses
