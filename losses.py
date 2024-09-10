# Based on SLIP code bases
# https://github.com/facebookresearch/SLIP
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import numpy as np

import utils

def get_metric_names():
    metics = ["loss"]
    metics.extend(["reconst_loss","ibot_patch_loss", "ibot_cls_loss", "ibot_loss", "contra_loss_1","contra_loss_2","clip_acc"])

    return metics


def cal_simsiam_loss(p, z, version="simplified"):  # negative cosine similarity
    if version == "original":
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif (
        version == "simplified"
    ):  # same thing, much faster. Scroll down, speed test in __main__
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class ACLIPLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.simclr_loss = SIMCLRLoss(temperature=temperature)

    def forward(self, outputs):
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]

        # cal simclr_loss
        bs = text_embed.shape[0]
        image_ssl_embed = outputs["image_ssl_embed"]
        inputs = {}
        inputs["aug1_embed"] = image_ssl_embed[:bs]
        inputs["aug2_embed"] = image_ssl_embed[bs:] # needs to be changed to "bs:"
        simclr_loss_dict = self.simclr_loss(inputs)

        def loss_fn(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1) # cosine similarity 

        im_features = outputs["byol_feats"]
        im_features_e = outputs["byol_feats_e"]
        im_features_e = torch.cat([im_features_e, im_features_e], dim=0)
        im_byol_loss = loss_fn(im_features, im_features_e).mean() # mean cos similarity loss across all samples

        local_batch_size = text_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size
        # assigns unique labels in dist setting. 
        # proc1 = [0, 1, 2, 3, 4]
        # proc2 = [5, 6, 7, 8, 9] and so on ...

        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        image_embed_1 = image_embed[:local_batch_size]
        image_embed_2 = image_embed[local_batch_size:]

        (
            image_embed_all_1,
            image_embed_all_2,
            text_embed_all,
        ) = utils.all_gather_batch_with_grad([image_embed_1, image_embed_2, text_embed])
        # image_embed_all_1 = [world_size * bs, dim]
        # image_embed_all_2 = [world_size * bs, dim]
        # text_embed_all = [world_size * bs, dim]

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed_1 @ text_embed_all.t() # [bs, world_size * bs]
        logits_per_text = logit_scale * text_embed @ image_embed_all_1.t() # [bs, world_size * bs]

        contra_loss_1 = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        logits_per_image = logit_scale * image_embed_2 @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all_2.t()

        contra_loss_2 = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2
       

        loss = (
            0.5 * contra_loss_1
            + 0.5 * contra_loss_2
            + simclr_loss_dict["ssl_loss"]
            + 2 * im_byol_loss
        )

        # compute accuracy (this is only doing it on image_embed_2's logits)
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {
            "loss": loss,
            "simclr_loss": simclr_loss_dict["ssl_loss"],
            "im_byol_loss": im_byol_loss,
            "contra_loss_1": contra_loss_1,
            "contra_loss_2": contra_loss_2,
            "clip_acc": acc,
        }
      

class DetailCLIPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.ibot_loss = iBOTLoss(nepochs=args.epochs)
        self.reconst_loss = reconstLoss(norm_pix_loss= not args.disable_norm_pix_loss, patch_size=args.patch_size)
        self.clip_loss_weight = args.clip_loss_weight
        self.ibot_patch_loss_weight = args.ibot_patch_loss_weight
        self.ibot_cls_loss_weight = args.ibot_cls_loss_weight
        self.reconst_loss_weight = args.reconst_loss_weight

    def forward(self, outputs, epoch):
        u = outputs["u"]
        v = outputs["v"]
        img_embed_u = outputs["img_embed_u"]
        img_embed_v = outputs["img_embed_v"]
        text_embed = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]

        local_batch_size = text_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=img_embed_u.device
            )
            self.last_local_batch_size = local_batch_size
        
        img_embed_u = F.normalize(img_embed_u, dim=-1, p=2)
        img_embed_v = F.normalize(img_embed_v, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # masks
        mask_u = outputs["mask_u"]
        mask_v = outputs["mask_v"]

        # cal ibot_loss
        student_ibot = outputs["student_ibot"]
        teacher_ibot = outputs["teacher_ibot"]
        all_ibot_loss = self.ibot_loss(student_ibot, teacher_ibot, None, [mask_u.view(-1,14,14), mask_v.view(-1,14,14)], epoch)

        # cal reconst_loss
        u_s_reconstructed = outputs["u_s_reconstructed"]
        v_s_reconstructed = outputs["v_s_reconstructed"]
        reconst_loss = 0.5 * (self.reconst_loss(u, u_s_reconstructed, mask_u) + self.reconst_loss(v, v_s_reconstructed, mask_v))


        (
            image_embed_all_u,
            image_embed_all_v,
            text_embed_all,
        ) = utils.all_gather_batch_with_grad([img_embed_u, img_embed_v, text_embed])
        

        # cosine similarity as logits
        logits_per_image = logit_scale * img_embed_u @ text_embed_all.t() # [bs, world_size * bs]
        logits_per_text = logit_scale * text_embed @ image_embed_all_u.t() # [bs, world_size * bs]

        contra_loss_1 = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        logits_per_image = logit_scale * img_embed_v @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all_v.t()

        contra_loss_2 = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size


        loss = (
            self.clip_loss_weight * (0.5 * contra_loss_1 + 0.5 * contra_loss_2)
            + self.reconst_loss_weight * reconst_loss
            + self.ibot_patch_loss_weight * all_ibot_loss['patch']
            + self.ibot_cls_loss_weight * all_ibot_loss['cls']
        )
        return {
            "loss": loss,
            "contra_loss_1": contra_loss_1,
            "contra_loss_2": contra_loss_2,
            "reconst_loss": reconst_loss,
            "ibot_patch_loss": all_ibot_loss['patch'],
            "ibot_cls_loss": all_ibot_loss['cls'],
            "ibot_loss": all_ibot_loss['loss'],
            "clip_acc": acc,
        }

        
        


        
      
class iBOTLoss(nn.Module):

    def __init__(self, nepochs, out_dim=8192, patch_out_dim=8192, ngcrops=2, nlcrops=0, warmup_teacher_temp=0.04, 
                 teacher_temp=0.04, warmup_teacher_temp2=0.04, teacher_temp2=0.07, 
                 warmup_teacher_temp_epochs=1, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        if dist.is_initialized():
            # The code to run in a distributed environment
            cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
            dist.all_reduce(cls_center)
            cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())

            patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
            dist.all_reduce(patch_center)
            patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        else:
            # The code to run on a single GPU or non-distributed setup
            cls_center = torch.sum(teacher_cls, dim=0, keepdim=True) / len(teacher_cls)
            patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True) / len(teacher_patch)

        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)


class reconstLoss(nn.Module):
    def __init__(self, norm_pix_loss=True, patch_size=16):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None 

    def forward(self, outputs):
        q_a = outputs["aug1_embed"]
        q_b = outputs["aug2_embed"]

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {"loss": loss, "ssl_loss": loss, "ssl_acc": acc}
