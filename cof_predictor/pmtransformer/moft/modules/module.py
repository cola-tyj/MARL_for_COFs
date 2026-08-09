# MOFTransformer version 2.1.0
import os
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
# from moftransformer.modules import heads, module_utils, objectives
# from moftransformer.modules.cgcnn import GraphEmbeddings
# from moftransformer.modules.module_utils import Normalizer
# from moftransformer.modules.vision_transformer_3d import VisionTransformer3D
from modules import heads, module_utils, objectives
from modules.cgcnn import GraphEmbeddings
from modules.module_utils import Normalizer
from modules.vision_transformer_3d import VisionTransformer3D
from pytorch_lightning import LightningModule
from sklearn.metrics import r2_score
from torch import Tensor


class Module(LightningModule):
    def __init__(self, config):
        """
        构建PMTransformer模型

        Args:
            config: 模型参数
        """
        super().__init__()
        
        # !!!!! 指定图嵌入的保存位置
        self.graphEmbedSavePath = config['graph_embed_save_path']
        
        self.save_hyperparameters()  # 使用pl保存设置的超参数

        self.max_grid_len = config["max_grid_len"]
        self.vis = config["visualize"]

        # 使用CGCNN提取图特征,作为输入的局部特征
        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
        )
        self.graph_embeddings.apply(objectives.init_weights)

        # 用于标识局部特征/全局特征
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # 构建ViT提取能量网格的特征,作为全局特征
        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        # CLS用于输出预测结果
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # VOL用于传递体积信息
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # 用于处理CLS,将其映射为向量,用于后续预测
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== 根据预训练任务构建不同的下游,将CLS pooler后的结果输入 =====================
        if config["loss_names"]["ggm"] > 0:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if config["loss_names"]["mtp"] > 0:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if config["loss_names"]["vfp"] > 0:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if config["loss_names"]["moc"] > 0 or config["loss_names"]["bbc"] > 0:
            self.moc_head = heads.MOCHead(config["hid_dim"])
            self.moc_head.apply(objectives.init_weights)

        # ===================== 最终执行分类或者回归 =====================
        hid_dim = config["hid_dim"]

        if config["load_path"] != "" and not config["test_only"]:
            # 加载模型参数
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.hparams.config["loss_names"]["regression"] > 0:
            # 回归任务下游
            self.regression_head = heads.RegressionHead(hid_dim)
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            # 分类任务下游
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        # 根据任务设置指标
        module_utils.set_metrics(self)
        self.current_tasks = list()

        # ========= load downstream (test_only) 加载下游网络的参数,仅用于测试环节 ===========

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True

    def infer(
        self,
        batch,
        mask_grid=False,
    ):
        """
        设置网络结构,构建返回值模版

        Args:
            batch (list): 当前的batch,包含B个晶体
            mask_grid (bool, optional): 在DeiT中使用以掩蔽图像块. Defaults to False.

        Returns:
            ret(dict): 多个任务的预测结果模版
        """
        # 读取配置
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']当前batch内所有原子核电荷数
        nbr_idx = batch["nbr_idx"]  # [N', M]存在连接的原子下标
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]原子间连接特征
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]经过类型、邻域分类后,不同的原子下标
        uni_count = batch["uni_count"]  # list [B]经过类型、邻域分类后,不同的原子个数

        grid = batch["grid"]  # [B, C(channel), H, W, D],能量网格
        volume = batch["volume"]  # list [B],体积列表

        # 如果包含moc或bbc(building block classification)的操作
        if "moc" in batch.keys():
            moc = batch["moc"]  # [B]
        elif "bbc" in batch.keys():
            moc = batch["bbc"]  # [B]
        else:
            moc = None

        # ============== 局部特征 ===============
        # 使用CGCNN获取图嵌入结果,作为局部特征
        # CGCNN执行完之后,再根据uni_idx和uni_count从每种原子中选择一个
        (
            graph_embeds,  # [B, max_graph_len, hid_dim],
            graph_masks,  # [B, max_graph_len],
            mo_labels,  # if moc: [B, max_graph_len], else: None
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
            cifIdList=cif_id
        )
        # NOTE:在此将嵌入结果写入文件,需要传入保存路径!!!!!
        # graphEmbedSavePath = "/home/zhangyi/dataset/cof_zx/graph_embed"
        saveGraphEmbedding(savePath=self.graphEmbedSavePath, cifIdList=cif_id, graphEmbeds=graph_embeds, graphMask=graph_masks)

        # 设置并拼接分类头CLS
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)  # [B, 1]

        graph_embeds = torch.cat(
            [cls_embeds, graph_embeds], dim=1
        )  # [B, max_graph_len+1, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, max_graph_len+1],标识原子是否被掩蔽

        # ============== 全局特征 ===============
        # 获取能量网格,因为包含SEP所以+1
        (
            grid_embeds,  # [B, max_grid_len+1, hid_dim]
            grid_masks,  # [B, max_grid_len+1]
            grid_labels,  # [B, grid+1, C] if mask_image == True
        ) = self.transformer.visual_embed(
            grid,
            max_image_len=self.max_grid_len,
            mask_it=mask_grid,
        )

        # 在能量网格中加入体积信息,+1->+2
        volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
        volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)
        grid_embeds = torch.cat(
            [grid_embeds, volume_embeds], dim=1
        )  # [B, max_grid_len+2, hid_dim]
        grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2]

        # 增加特征类型信息,标识全局特征/局部特征
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks, device=self.device).long()
        )
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks, device=self.device).long()
        )

        # 拼接局部特征和全局特征,整体作为transformer的输入
        co_embeds = torch.cat(
            [graph_embeds, grid_embeds], dim=1
        )  # [B, final_max_len, hid_dim]
        co_masks = torch.cat(
            [graph_masks, grid_masks], dim=1
        )  # [B, final_max_len]

        # === transformer-block -> norm -> (图特征, CLS特征...) ===
        x = co_embeds
        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

            if self.vis:
                attn_weights.append(_attn)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels,  # if MPP, else None
            "mo_labels": mo_labels,  # if MOC, else None
            "cif_id": cif_id,
            "attn_weights": attn_weights,
        }

        return ret

    def forward(self, batch):
        """
        执行

        Args:
            batch (list): batch内的晶体

        Returns:
            ret(dict): 多个任务的预测结果
        """
        ret = dict()

        # 当前不包含任务,则仅执行infer的操作
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        # 否则根据指定任务,将输入带有下游网络的结构中

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Graph Grid Matching
        if "ggm" in self.current_tasks:
            ret.update(objectives.compute_ggm(self, batch))

        # MOF Topology Prediction
        if "mtp" in self.current_tasks:
            ret.update(objectives.compute_mtp(self, batch))

        # Void Fraction Prediction
        if "vfp" in self.current_tasks:
            ret.update(objectives.compute_vfp(self, batch))

        # Metal Organic Classification (or Building Block Classfication)
        if "moc" in self.current_tasks or "bbc" in self.current_tasks:
            ret.update(objectives.compute_moc(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = Normalizer(self.mean, self.std)
            ret.update(objectives.compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch))
        return ret

    
    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self)

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        module_utils.epoch_wrapup(self)

    def on_test_start(self,):
        module_utils.set_task(self)
    
    def test_step(self, batch, batch_idx):
        output = self(batch)
        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        }  # update cpu for memory

        if 'regression_logits' in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()
        return output

    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if len(self.test_logits) > 1:
            r2 = r2_score(
                np.array(self.test_labels), np.array(self.test_logits)
            )
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.test_labels.clear()
            self.test_logits.clear()

    def configure_optimizers(self):
        return module_utils.set_schedule(self)
    
    def on_predict_start(self):
        self.write_log = False
        module_utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        
        if 'classification_logits' in output:
            if self.hparams.config['n_classes'] == 2:
                output['classification_logits_index'] = torch.round(output['classification_logits']).to(torch.int)
            else:
                softmax = torch.nn.Softmax(dim=1)
                output['classification_logits'] = softmax(output['classification_logits'])
                output['classification_logits_index'] = torch.argmax(output['classification_logits'], dim=1)

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ('logits' in k) or ('labels' in k) or 'cif_id' == k
        }

        return output
    
    def on_predict_epoch_end(self, *args):
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(self, ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        """
        优化器执行参数优化

        Args:
            scheduler: 优化器
        """
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            metric, = args
        else:
            raise ValueError('lr_scheduler_step must have metric and optimizer_idx(optional)')

        if pl.__version__ >= '2.0.0':
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()

def saveGraphEmbedding(savePath: str, cifIdList: list, graphEmbeds: Tensor, graphMask: Tensor) -> None:
    """
    按照Batch保存CGCNN的预嵌入结果

    Args:
        savePath(str): 保存路径,每个CIF对应两个文件
        cifIdList (list): CIF名
        graphEmbeds (Tensor): 嵌入结果[B, max_graph_len, hid_dim]
        graphMask (Tensor): 掩蔽[B, max_graph_len]
    """
    os.makedirs(savePath, exist_ok=True)
    for i, cif in enumerate(cifIdList):
        feature = graphEmbeds[i].cpu().numpy()
        mask = graphMask[i].cpu().numpy()
        np.save(f"{savePath}/{cif}_graph.npy", feature)
        np.save(f"{savePath}/{cif}_mask.npy", mask)