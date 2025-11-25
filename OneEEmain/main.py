import argparse
import random

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from model import Model
from utils import get_device, move_to_device


def _to_cpu(x):
    return x.detach().float().cpu()

def _as_dict(results):
    """
    Chuẩn hoá mọi kiểu trả về của model thành dict để dễ debug.
    - Nếu model trả dict thì dùng luôn.
    - Nếu model trả tuple/list thì map [0]=tri, [1]=arg, [2]=role (đoán).
    """
    if isinstance(results, dict):
        return results
    if isinstance(results, (tuple, list)):
        out = {}
        if len(results) >= 1: out["tri_logits"]  = results[0]
        if len(results) >= 2: out["arg_logits"]  = results[1]
        if len(results) >= 3: out["role_logits"] = results[2]
        return out
    return {"logits": results}

# def _print_tensor_stat(name, t):
#     try:
#         t = _to_cpu(t)
#         print(
#             f"[STAT] {name:12s} shape={tuple(t.shape)}  "
#             f"min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}"
#         )
#     except Exception as e:
#         print(f"[STAT] {name:12s} <cannot print> ({e})")

def debug_dump_outputs(model, loader, device, thr: float = 0.3, topk: int = 5):
    model.eval()
    batch = next(iter(loader))

    (inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d,
     tri_labels, arg_labels, role_labels, event_idx, tuple_labels, role_label_num) = batch

    inputs        = inputs.to(device)
    att_mask      = att_mask.to(device)
    word_mask1d   = word_mask1d.to(device)
    word_mask2d   = word_mask2d.to(device)
    triu_mask2d   = triu_mask2d.to(device)
    tri_labels    = tri_labels.to(device)
    arg_labels    = arg_labels.to(device)
    role_labels   = role_labels.to(device)

    torch.cuda.empty_cache()
    use_amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=use_amp_dtype):
        results = model(
            inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d,
            tri_labels, arg_labels, role_labels
        )

    # ("\n[DEBUG] ===== MODEL RAW OUTPUT =====")
    # print("type(resuprintlts):", type(results))

    # # Trường hợp 1: model trả dict
    # if isinstance(results, dict):
    #     for k, v in results.items():
    #         if hasattr(v, "shape"):
    #             print(f"[RES] {k}: type={type(v)}, shape={v.shape}")
    #         else:
    #             print(f"[RES] {k}: type={type(v)}")

    #     # thử xem 1 vài phần tử đầu nếu là numpy / tensor
    #     for k in ["ti", "tc", "ai", "ac"]:
    #         if k in results:
    #             arr = results[k]
    #             try:
    #                 import numpy as np
    #                 if isinstance(arr, np.ndarray):
    #                     print(f"[RES] {k} sample:", arr[:3])
    #                 elif hasattr(arr, "detach"):
    #                     print(f"[RES] {k} sample:", arr.detach().cpu().numpy()[:3])
    #             except Exception as e:
    #                 print(f"[RES] {k} print error:", e)

    # # Trường hợp 2: model trả tuple/list
    # elif isinstance(results, (tuple, list)):
    #     for idx, v in enumerate(results):
    #         name = f"out[{idx}]"
    #         if hasattr(v, "shape"):
    #             print(f"[RES] {name}: type={type(v)}, shape={v.shape}")
    #         else:
    #             print(f"[RES] {name}: type={type(v)}")

def compute_kl_loss(p, q):
    p_loss = F.kl_div(p, q, reduction='none')
    q_loss = F.kl_div(q, p, reduction='none')

    # pad_mask is for seq-level tasks
    # if pad_mask is not None:
    #     p_loss.masked_fill_(pad_mask, 0.)
    #     q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # p_loss = p_loss.sum()
    # q_loss = q_loss.sum()
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def compute_dis_loss(x):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.sum((x - x_mean) ** 2, dim=-1) / x.size(-1)
    loss = torch.sqrt(var)
    loss = loss.mean()
    return loss


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_true = y_true.float().detach()
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class Trainer(object):
    def __init__(self, model):
        self.model = model

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=updates_total
        )

        # BCE cho multi-label
        self.bce_tri  = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_arg  = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_role = nn.BCEWithLogitsLoss(reduction="none")

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        total_tc_r = total_tc_p = total_tc_c = 0
        total_ai_r = total_ai_p = total_ai_c = 0

        for i, data_batch in enumerate(data_loader):
            data_batch = [d.cuda() for d in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]
            inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, \
            tri_labels, arg_labels, role_labels, event_idx, _, _ = data_batch

            tri_logits, arg_logits, role_logits = self.model(
                inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d,
                tri_labels, arg_labels, role_labels, event_idx
            )
            # role_logits = None

            # ===== reshape labels =====
            tri_labels = tri_labels.permute(0, 2, 3, 1).float()   # [B,L,L,E]
            arg_labels = arg_labels.permute(0, 2, 3, 1).float()
            # role_labels = role_labels.permute(0, 2, 3, 1, 4).float()  # [B,L,L,E,R]
            # role_labels: [B,L,L,E,R] -> [B,L,L,R] bằng cách max theo E
            role_labels = role_labels.max(dim=3).values    # [B,L,L,R]


            mask_tri = word_mask2d.unsqueeze(-1).float()
            mask_arg = word_mask2d.unsqueeze(-1).float()
            role_arg = triu_mask2d.unsqueeze(-1).float()

            tri_loss_raw = self.bce_tri(tri_logits, tri_labels)
            tri_pos_mask = tri_labels.eq(1.0)
            tri_loss_raw = tri_loss_raw * (1.0 + 9.0 * tri_pos_mask.float())
            tri_loss = (tri_loss_raw * mask_tri).sum() / mask_tri.sum().clamp_min(1.0)

            arg_loss_raw = self.bce_arg(arg_logits, arg_labels)
            arg_pos_mask = arg_labels.eq(1.0)
            arg_loss_raw = arg_loss_raw * (1.0 + 7.0 * arg_pos_mask.float())
            arg_loss = (arg_loss_raw * mask_arg).sum() / mask_arg.sum().clamp_min(1.0)

            # Role loss (mới)
            # role_loss_raw = self.bce_role(role_logits, role_labels)   # [B,L,L,R]
            # role_pos_mask = role_labels.eq(1.0)
            # role_loss_raw = role_loss_raw * (1.0 + 4.0 * role_pos_mask.float())
            # role_loss = (role_loss_raw * role_arg).sum() / role_arg.sum().clamp_min(1.0)

            loss = config.gamma * tri_loss + arg_loss
            # loss = config.gamma * tri_loss + arg_loss   # KHÔNG CÒN role_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            loss_list.append(loss.detach().cpu().item())

            # Train F1
            tri_outputs = (torch.sigmoid(tri_logits) > 0.5)
            total_tc_r += tri_labels.long().sum().item()
            total_tc_p += tri_outputs.sum().item()
            total_tc_c += (tri_outputs & tri_labels.bool()).sum().item()

            arg_outputs = (torch.sigmoid(arg_logits) > 0.5)
            total_ai_r += arg_labels.long().sum().item()
            total_ai_p += arg_outputs.sum().item()
            total_ai_c += (arg_outputs & arg_labels.bool()).sum().item()

            # role_outputs = (torch.sigmoid(role_logits) > 0.5)
            # total_ac_r += role_labels.long().sum().item()
            # total_ac_p += role_outputs.sum().item()
            # total_ac_c += (role_outputs & role_labels.bool()).sum().item()


        tri_f1, tri_r, tri_p = utils.calculate_f1(total_tc_r, total_tc_p, total_tc_c)
        arg_f1, arg_r, arg_p = utils.calculate_f1(total_ai_r, total_ai_p, total_ai_c)
        # role_f1, role_r, role_p = utils.calculate_f1(total_ac_r, total_ac_p, total_ac_c)

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "Tri F1", "Arg F1", "Role F1"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                    ["{:3.4f}".format(x) for x in [tri_f1, arg_f1, 0.0]])  # Role F1 = 0
        logger.info("\n{}".format(table))

        return tri_f1 + arg_f1


    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        total_results = {k + "_" + t: 0 for k in ["ti", "tc", "ai", "ac"] for t in ["r", "p", "c"]}

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                cuda_part = [d.cuda() for d in data_batch[:-2]]
                inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, \
                tri_labels, arg_labels, role_labels, event_idx = cuda_part
                tuple_labels, _ = data_batch[-2], data_batch[-1]

                results = self.model(
                    inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d,
                    tri_labels, arg_labels, role_labels, event_idx=None
                )
                results = utils.decode(results, tuple_labels, config.tri_args)
                for key, value in results.items():
                    total_results[key] += value

        ti_f1, ti_r, ti_p = utils.calculate_f1(total_results["ti_r"], total_results["ti_p"], total_results["ti_c"])
        tc_f1, tc_r, tc_p = utils.calculate_f1(total_results["tc_r"], total_results["tc_p"], total_results["tc_c"])
        ai_f1, ai_r, ai_p = utils.calculate_f1(total_results["ai_r"], total_results["ai_p"], total_results["ai_c"])
        ac_f1, ac_r, ac_p = utils.calculate_f1(total_results["ac_r"], total_results["ac_p"], total_results["ac_c"])

        title = "EVAL" if not is_test else "TEST"
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Trigger I"] + ["{:3.4f}".format(x) for x in [ti_f1, ti_p, ti_r]])
        table.add_row(["Trigger C"] + ["{:3.4f}".format(x) for x in [tc_f1, tc_p, tc_r]])
        table.add_row(["Argument I"] + ["{:3.4f}".format(x) for x in [ai_f1, ai_p, ai_r]])
        table.add_row(["Argument C"] + ["{:3.4f}".format(x) for x in [ac_f1, ac_p, ac_r]])
        logger.info("\n{}".format(table))

        return (ti_f1 + ai_f1 + tc_f1 + ac_f1) / 4


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/bkee.json')
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--tri_hid_size', type=int)
    parser.add_argument('--eve_hid_size', type=int)
    parser.add_argument('--arg_hid_size', type=int)
    parser.add_argument('--node_type_size', type=int)
    parser.add_argument('--event_sample', type=int)
    parser.add_argument('--layers', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--graph_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warm_epochs', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--grad_clip_norm', type=float)
    parser.add_argument('--gamma', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    if config.seed >= 0:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets = data_loader.load_data(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=2,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    warmup_steps = config.warm_epochs * len(datasets[0])

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)

        # 🔍 DEBUG: In output model sau epoch đầu tiên
        # if i == 0:
        #     logger.info("[DEBUG] Dumping model outputs after epoch 0...")
        #     try:
        #         model = trainer.model
        #     except:
        #         model = trainer.net  # phòng trường hợp trainer đặt tên khác

        #     try:
        #         device = trainer.device
        #     except:
        #         device = next(model.parameters()).device

        #     # gọi hàm debug
        #     debug_dump_outputs(model, dev_loader, device, thr=0.3, topk=10)

        # ---- phần đánh giá từ epoch >= 5 giữ nguyên ----
        if i >= 5:
            f1 = trainer.eval(i, dev_loader)
            test_f1 = trainer.eval(i, test_loader, is_test=True)
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                trainer.save("model.pt")

    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("model.pt")
    trainer.eval("Final", test_loader, True)


