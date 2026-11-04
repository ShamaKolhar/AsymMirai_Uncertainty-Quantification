import torch
import torchvision.models as models
import sys
sys.path.append('../')

def exam_collate_function(batch):
    eids = [b.eid for b in batch]
    return batch


# Define Dropout-enabled ResNet18 backbone
class DropoutResNet18(torch.nn.Module):
    def __init__(self, base_model, p=0.7):
        super().__init__()
        self.p = p
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.dropout = torch.nn.Dropout(p=self.p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        return x


class LocalizedDifModel(torch.nn.Module):
    def __init__(self,
                 alignment_space=None,
                 asymmetry_metric=None,
                 latent_h=5,
                 latent_w=5,
                 embedding_channel=512,
                 embedding_model=None,
                 use_stretch=False,
                 train_backbone=True,
                 initial_asym_mean=8000000,
                 initial_asym_std=1520381,
                 flexible_asymmetry=False,
                 use_stretch_matrix=False,
                 device_ids=[0],
                 use_addon_layers=False,
                 topk_for_heatmap=None,
                 use_bias=False,
                 use_bn=False,
                 dropout_prob=0.9):
        super().__init__()
        self.initial_asym_mean = initial_asym_mean
        self.initial_asym_std = initial_asym_std
        self.use_bn = use_bn
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        if use_bn:
            print("Using batch norm instead of learned stats")
            self.use_bias = False
            self.bn = torch.nn.BatchNorm2d(embedding_channel).cuda()
        elif use_bias:
            self.learned_asym_mean = torch.tensor(initial_asym_mean).float().cuda()
            self.learned_asym_mean.requires_grad = True
            self.learned_asym_std = torch.tensor(initial_asym_std).float().cuda()
            self.learned_asym_std.requires_grad = True

        # --------- Backbone setup ---------
        if embedding_model is not None:
            self.backbone = embedding_model
        else:
            base_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = DropoutResNet18(base_resnet, p=dropout_prob)

        self.backbone = torch.nn.DataParallel(self.backbone, device_ids=device_ids).cuda()
        for param in self.backbone.parameters():
            param.requires_grad = True

        if use_addon_layers:
            print("USING ADDON LAYERS")
            self.conv1 = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.conv1.requires_grad = True

        self.train_backbone = train_backbone
        self.asymmetry_metric = asymmetry_metric if asymmetry_metric is not None else self.fallback_asym
        self.flexible_asymmetry = flexible_asymmetry
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.topk_for_heatmap = topk_for_heatmap

        if self.topk_for_heatmap is not None:
            self.topk_weights = (torch.arange(self.topk_for_heatmap) /
                                 torch.sum(torch.arange(self.topk_for_heatmap))).float().cuda()
            self.topk_weights.requires_grad = True

        self.alignment_space = alignment_space
        self.use_stretch = use_stretch
        self.use_stretch_matrix = use_stretch_matrix

        if use_stretch_matrix:
            self.cc_stretch_params = torch.eye(embedding_channel).float().cuda()
            self.cc_stretch_params.requires_grad = True
            self.mlo_stretch_params = torch.eye(embedding_channel).float().cuda()
            self.mlo_stretch_params.requires_grad = True
        else:
            self.cc_stretch_params = torch.ones(embedding_channel).float().cuda()
            self.cc_stretch_params.requires_grad = True
            self.mlo_stretch_params = torch.ones(embedding_channel).float().cuda()
            self.mlo_stretch_params.requires_grad = True

    def fallback_asym(self, left, right, **kwargs):
        return torch.norm(left - right)

    def stretched_asymmetries(self, left, right, mode="MLO"):
        if mode == "MLO":
            left_stretched = self.mlo_stretch_params.view(1, -1, 1, 1) * left
            right_stretched = self.mlo_stretch_params.view(1, -1, 1, 1) * right
        else:
            left_stretched = self.cc_stretch_params.view(1, -1, 1, 1) * left
            right_stretched = self.cc_stretch_params.view(1, -1, 1, 1) * right
        return torch.norm(left_stretched - right_stretched, dim=-1)

    def matrix_stretch(self, embedding, mode="MLO"):
        shape = embedding.shape
        if mode == "MLO":
            stretched = (self.mlo_stretch_params @ embedding.transpose(0, 1).reshape(shape[1], -1)).view(
                shape[1], shape[0], shape[2], shape[3]).transpose(0, 1)
        else:
            stretched = (self.cc_stretch_params @ embedding.transpose(0, 1).reshape(shape[1], -1)).view(
                shape[1], shape[0], shape[2], shape[3]).transpose(0, 1)
        return stretched

    def forward(self, left_cc, right_cc, left_mlo, right_mlo, exam_list=None, **kwargs):
        if exam_list is None:
            exam_list = [(left_cc, right_cc, 'CC'), (left_mlo, right_mlo, 'MLO')]

        mean_asymmetry = 0
        count_list = torch.ones(exam_list[0][0].shape[0]) * len(exam_list)
        count_list.requires_grad = False
        other_returns = []

        for left, right, view in exam_list:
            sub_list = torch.zeros_like(count_list)
            sub_list.requires_grad = False
            sub_list[torch.sum(left, dim=(1, 2, 3)).cpu() == 0] = 1
            count_list = count_list - sub_list

            if self.alignment_space == 'pixel':
                right = torch.flip(right, dims=[-1])

            with torch.set_grad_enabled(self.train_backbone):
                left_embedding = self.backbone(left)
                right_embedding = self.backbone(right)

            if self.alignment_space == 'latent':
                right_embedding = torch.flip(right_embedding, dims=[-1])

            left_embedding = self.dropout(left_embedding)
            right_embedding = self.dropout(right_embedding)

            if self.use_stretch:
                if self.use_stretch_matrix:
                    left_embedding = self.matrix_stretch(left_embedding, mode=view)
                    right_embedding = self.matrix_stretch(right_embedding, mode=view)
                else:
                    if 'CC' in view:
                        left_embedding *= self.cc_stretch_params.view(1, -1, 1, 1)
                        right_embedding *= self.cc_stretch_params.view(1, -1, 1, 1)
                    else:
                        left_embedding *= self.mlo_stretch_params.view(1, -1, 1, 1)
                        right_embedding *= self.mlo_stretch_params.view(1, -1, 1, 1)

            if self.use_bn:
                left_embedding = self.bn(left_embedding)
                right_embedding = self.bn(right_embedding)

            asymmetry, other = self.asymmetry_metric(left_embedding, right_embedding,
                                                     latent_h=self.latent_h,
                                                     latent_w=self.latent_w,
                                                     flexible=self.flexible_asymmetry,
                                                     topk=self.topk_for_heatmap)

            if self.topk_for_heatmap is not None:
                asymmetry = asymmetry @ self.topk_weights.view(-1, 1)
                asymmetry = asymmetry.view(-1)

            other_returns.append(other)
            mean_asymmetry += (asymmetry * (1 - sub_list).cuda())

        count_list = count_list.cuda()
        if not self.use_bn:
            scaled_asym = ((mean_asymmetry / count_list) - self.initial_asym_mean) / self.initial_asym_std
        else:
            scaled_asym = mean_asymmetry / count_list

        pred = torch.sigmoid(scaled_asym)
        return torch.stack([1 - pred, pred], dim=1), other_returns
