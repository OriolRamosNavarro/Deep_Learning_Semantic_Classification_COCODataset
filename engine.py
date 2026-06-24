import torch
from tqdm.auto import tqdm


def entrenar_una_epoca(model, loader, optimizer, criterion, device,
                       scaler=None, scheduler=None, use_amp=False,
                       channels_last=False, grad_clip_norm=0.0, epoch=None,
                       ema=None, metrics=None):
    """
    EXPLICACIÓ SIMPLE: Entrena el model durant un epoch (una passada per totes les dades).
    Per a cada batch:
    1. Forward del model (en mixed precision fp16 si use_amp=True)
    2. Calcula la pèrdua en fp32 (la Dice usa smooth=1e-6 que es perd en fp16)
    3. Backward + (opcional) recorte de gradiente + step (amb GradScaler si AMP)
    4. (Opcional) avança el scheduler per batch — necessari per al warmup
    5. (Opcional) actualitza l'EMA dels pesos del model
    6. (Opcional) acumula la confusion matrix de train si es passa `metrics`
       (per loguear train_mIoU i poder veure overfitting). El cridador llegeix
       el resultat amb metrics.calcular() després d'aquesta funció.
    Retorna la pèrdua promitjada de l'epoch.
    """
    model.train()
    total_loss = 0.0
    if metrics is not None:
        metrics.reinicialitzar()
    desc = f"train ep{epoch:03d}" if epoch is not None else "train"
    pbar = tqdm(loader, desc=desc, leave=False)
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    amp_active = use_amp and scaler is not None and scaler.is_enabled()

    for images, masks in pbar:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
        loss = criterion(preds.float(), masks)   # loss siempre en fp32
        if metrics is not None:
            metrics.actualitzar(preds.detach(), masks)   # mIoU de train (argmax + bincount)

        if amp_active:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()   # per-batch step (warmup + cosine annealing)

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validar(model, loader, criterion, metrics, device,
            use_amp=False, channels_last=False, epoch=None):
    """
    EXPLICACIÓ SIMPLE: Avalua el model en dades de validació (sense actualitzar pesos).
    Per a cada batch: forward, calcula la pèrdua en fp32 i acumula les mètriques.
    Retorna la pèrdua promitjada i les mètriques calculades (mIoU, IoU per classe).
    """
    model.eval()
    metrics.reinicialitzar()
    total_loss = 0.0
    desc = f"val   ep{epoch:03d}" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, leave=False)
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    for images, masks in pbar:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        masks  = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
        total_loss += criterion(preds.float(), masks).item()
        metrics.actualitzar(preds, masks)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / max(len(loader), 1), metrics.calcular()
