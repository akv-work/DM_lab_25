"""
Лабораторная работа №4. Послойное предобучение глубокой сети стеком ограниченных машин Больцмана
(RBM) по алгоритму из лекции (Contrastive Divergence, CD-1).

Для каждой RBM (видимый слой v размера n, скрытый слой h размера m, матрица W n×m,
пороги видимых нейронов b и скрытых c) на мини-батче:
    h1 = f_hid(v1·W + c)                      P(h = 1 | v1) для логистических нейронов
    h_s = h1 > rand   (логистические)  |  h_s = h1   (остальные активации)   <- как в лекции
    v2 = f_vis(h_s·Wᵀ + b)                    реконструкция
    h2 = f_hid(v2·W + c)
    ΔW = α·(v1ᵀ·h1 − v2ᵀ·h2) / batch          положительная фаза − отрицательная фаза
    Δb = α·mean(v1 − v2),  Δc = α·mean(h1 − h2)
Как в лекции: W₀ = 0.1·randn, пороги видимых нейронов = 0, моментум 0.5 в первых 5 эпохах, затем 0.9;
условие останова — max_epochs или ошибка реконструкции < min_error. Функции активации RBM берутся
из сети (act_func_vis, act_func_hid в лекции): скрытые нейроны RBM — той же активации, что и слой сети,
видимые нейроны первой RBM линейные (вещественные стандартизованные признаки), следующих — активации
предыдущего слоя. Пороги скрытых нейронов инициализируются значением −1 (как в лекции) для логистических
нейронов и 0 для остальных (при ReLU и пороге −1 большинство нейронов «умирает» и не обучается).
Выход обученной RBM служит входом следующей. После обучения веса и пороги скрытых нейронов
переносятся в слои сети (W → nn.Linear.weight, c → bias), параметры видимых нейронов отбрасываются.
"""
import torch

from deep_net import linear_layers, batch_size_for, _batches, ACTIVATIONS


@torch.no_grad()
def pretrain_rbm(model, X, act_name, cfg, seed):
    """Предобучение скрытых слоёв model стеком RBM. Возвращает кривые ошибки реконструкции
    (средний квадрат ошибки на один элемент) по эпохам для каждой RBM."""
    gen = torch.Generator().manual_seed(seed + 3000)
    batch = batch_size_for(len(X))
    f_hid = ACTIVATIONS[act_name]()
    logistic = act_name == "sigmoid"
    H = X
    curves = []
    for li, lin in enumerate(linear_layers(model)[:-1]):
        n_vis, n_hid = lin.in_features, lin.out_features
        f_vis = (lambda z: z) if li == 0 else f_hid
        rates = cfg.rbm_rates[act_name]
        rate = rates[min(li, len(rates) - 1)]

        W = 0.1 * torch.randn(n_vis, n_hid, generator=gen)
        b = torch.zeros(n_vis)
        c = -torch.ones(n_hid) if logistic else torch.zeros(n_hid)
        vW, vb, vc = torch.zeros_like(W), torch.zeros_like(b), torch.zeros_like(c)

        curve = []
        for epoch in range(cfg.rbm_epochs):
            momentum = cfg.rbm_start_momentum if epoch < 5 else cfg.rbm_final_momentum
            err = 0.0
            for idx in _batches(len(H), batch, gen):
                v1 = H[idx]
                h1 = f_hid(v1 @ W + c)
                h_s = (h1 > torch.rand(h1.shape, generator=gen)).float() if logistic else h1
                v2 = f_vis(h_s @ W.T + b)
                h2 = f_hid(v2 @ W + c)

                n = len(idx)
                vW = vW * momentum + rate * ((v1.T @ h1 - v2.T @ h2) / n - cfg.rbm_weight_decay * W)
                vb = vb * momentum + rate * (v1 - v2).sum(0) / n
                vc = vc * momentum + rate * (h1 - h2).sum(0) / n
                W += vW
                b += vb
                c += vc
                err += ((v2 - v1) ** 2).sum().item()
            curve.append(err / (len(H) * n_vis))
            if err / len(H) < cfg.rbm_min_error:
                break
        curves.append(curve)

        lin.weight.copy_(W.T)      # веса RBM -> слой сети
        lin.bias.copy_(c)          # пороги скрытых нейронов -> смещения слоя
        H = f_hid(H @ W + c)       # вход следующей RBM
    return curves
