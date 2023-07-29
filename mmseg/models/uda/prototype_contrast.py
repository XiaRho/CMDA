import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class Prototype:
    def __init__(self, num_classes=19, features_dim=256, proto_momentum=0.9999):
        self.prototype = dict()
        self.prototype_num = dict()

        self.prototype['S_image'] = torch.zeros([num_classes, features_dim]).cuda()
        self.prototype['S_events'] = torch.zeros([num_classes, features_dim]).cuda()
        self.prototype['S_fusion'] = torch.zeros([num_classes, features_dim]).cuda()
        self.prototype['T_image'] = torch.zeros([num_classes, features_dim]).cuda()
        self.prototype['T_events'] = torch.zeros([num_classes, features_dim]).cuda()
        self.prototype['T_fusion'] = torch.zeros([num_classes, features_dim]).cuda()

        self.prototype_num['S_image'] = torch.zeros([num_classes]).cuda()
        self.prototype_num['S_events'] = torch.zeros([num_classes]).cuda()
        self.prototype_num['S_fusion'] = torch.zeros([num_classes]).cuda()
        self.prototype_num['T_image'] = torch.zeros([num_classes]).cuda()
        self.prototype_num['T_events'] = torch.zeros([num_classes]).cuda()
        self.prototype_num['T_fusion'] = torch.zeros([num_classes]).cuda()

        self.proto_momentum = proto_momentum
        self.num_classes = num_classes
        self.features_dim = features_dim

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.num_classes + 1, w, h).cuda()
        id = torch.where(label < self.num_classes, label, torch.Tensor([self.num_classes]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_prototype(self, feat_cls, outputs, labels_val=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        # convert to one-hot format: [batch, self.class_numbers + 1, w, h]
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)  # [batch, self.class_numbers + 1, 1, 1]
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.num_classes):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def update_all_prototype(self, features, outputs, keys):
        for key in keys:
            vectors, ids = self.calculate_prototype(features[key], outputs[key], labels_val=None)
            for t in range(len(ids)):
                self.update_single_prototype(ids[t], vectors[t].detach(), key=key, name='moving_average', start_mean=True)

    def update_single_prototype(self, id, vector, key, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.prototype_num[key][id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.prototype[key][id] = self.prototype[key][id] * (1 - self.proto_momentum) + self.proto_momentum * vector.squeeze()
            self.prototype_num[key][id] += 1
            self.prototype_num[key][id] = min(self.prototype_num[key][id], 3000)
        elif name == 'mean':
            self.prototype[key][id] = self.prototype[key][id] * self.prototype_num[key][
                id] + vector.squeeze()
            self.prototype_num[key][id] += 1
            self.prototype[key][id] = self.prototype[key][id] / self.prototype_num[key][id]
            self.prototype_num[key][id] = min(self.prototype_num[key][id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, contrast_config):
        super(PixelContrastLoss, self).__init__()

        self.temperature = contrast_config['temperature']
        self.base_temperature = contrast_config['base_temperature']
        self.ignore_label = contrast_config['ignore_label']
        self.max_samples = contrast_config['max_samples']
        self.max_views = contrast_config['max_views']

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]  # 当前batch_size中，所含类别数

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise ValueError('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.unsqueeze(1).float().clone()  # (B, 1, H, W)
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()  # (B, H, W)
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)  # (B, H*W)
        predict = predict.contiguous().view(batch_size, -1)  # (B, H*W)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # (B, H*W, 256)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss


class ContrastCELoss(nn.Module, ABC):
    def __init__(self, contrast_config):
        super(ContrastCELoss, self).__init__()

        self.loss_weight = contrast_config['loss_weight']
        self.contrast_criterion = PixelContrastLoss(contrast_config)
        self.pixel_update_freq = contrast_config['pixel_update_freq']
        self.memory_size = contrast_config['memory_size']

        self.register_buffer("segment_queue", torch.randn(contrast_config['num_classes'], self.memory_size, contrast_config['dim']))  # (19, 5000, 256)
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(contrast_config['num_classes'], dtype=torch.long))

        self.register_buffer("pixel_queue", torch.randn(contrast_config['num_classes'], self.memory_size, contrast_config['dim']))  # (19, 5000, 256)
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(contrast_config['num_classes'], dtype=torch.long))

    def _dequeue_and_enqueue(self, keys, labels):

        labels = labels[:, 0, :, :]
        image_keys, events_keys = keys[0], keys[1]
        keys = torch.cat((image_keys, events_keys), dim=0)

        segment_queue = self.segment_queue
        segment_queue_ptr = self.segment_queue_ptr
        pixel_queue = self.pixel_queue
        pixel_queue_ptr = self.pixel_queue_ptr

        # key: features_256.detach()
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]  # 256

        assert labels.shape[1] // keys.shape[2] == labels.shape[2] // keys.shape[3]
        self.network_stride = labels.shape[1] // keys.shape[2]  # 4

        labels = labels[:, ::self.network_stride, ::self.network_stride]  # seem to resize
        labels = torch.cat((labels, labels), dim=0)

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)  # (256, H*W)
            this_label = labels[bs].contiguous().view(-1)  # (H*W)
            this_label_ids = torch.unique(this_label)  # (19)
            this_label_ids = [x for x in this_label_ids if x > 0 and x != 255]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)  # 10
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    # pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % self.memory_size
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + K) % self.memory_size

    def forward(self, preds, target):
        image_seg = preds['image_output']
        events_seg = preds['events_output']
        seg = torch.cat((image_seg, events_seg), dim=0)  # 在batch维度cat在一起，即将image和events混合在一起

        image_embedding = preds['image_proj_feat']
        events_embedding = preds['events_proj_feat']
        embedding = torch.cat((image_embedding, events_embedding), dim=0)

        target = torch.cat((target, target), dim=0)[:, 0, :, :]

        '''
        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        if "pixel_queue" in preds:
            pixel_queue = preds['pixel_queue']
        else:
            pixel_queue = None
        '''
        segment_queue = self.segment_queue
        pixel_queue = self.pixel_queue

        assert segment_queue is not None and pixel_queue is not None
        queue = torch.cat((segment_queue, pixel_queue), dim=1)
        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict, queue)

        return self.loss_weight * loss_contrast

