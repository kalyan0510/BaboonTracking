import numpy as np
import cv2 as cv

from ps_hepers.helpers import imshow, non_max_suppression


class ShuffleObjectAugmentor:

    def __init__(self, max_aug=3, max_obj_shuffle=3, downscale_ratio=0.1):
        self.max_obj_shuffle = max_obj_shuffle
        self.downscale_ratio = downscale_ratio
        self.max_aug = max_aug

    def to_ltrb(self, a1):
        return np.asarray([a1[1] - a1[3] / 2, a1[2] - a1[4] / 2, a1[1] + a1[3] / 2, a1[2] + a1[4] / 2])

    def get_inner_outer_boundaries(self, shp, anno, padding=None):
        l, t, r, b = (self.to_ltrb(anno) * np.asarray([shp[1], shp[0], shp[1], shp[0]])).astype(np.int)
        W = [padding, int(1 / self.downscale_ratio)][padding is None]
        l_, t_, r_, b_ = max(0, l - W), max(0, t - W), min(shp[1], r + W), min(shp[0], b + W)
        return (l, t, r, b), (l_, t_, r_, b_)

    def get_annotation_neighbourhood_avg(self, image, anno):
        (l, t, r, b), (l_, t_, r_, b_) = self.get_inner_outer_boundaries(image.shape, anno)
        # imshow(image[t:b,l:r])
        return np.asarray(np.concatenate([image[t_:b_, l_:l].mean(axis=(0, 1)), image[t_:t, l_:r_].mean(axis=(0, 1)),
                                          image[t_:b_, r:r_].mean(axis=(0, 1)), image[b:b_, l_:r_].mean(axis=(0, 1))],
                                         axis=0)), (b_ - t_, r_ - l_)

    def get_all_annotations_neighbourhood_avg(self, img, annotations):
        return [self.get_annotation_neighbourhood_avg(img, anno) for anno in annotations]

    def find_best_shuffle_pos(self, downscaled_img, neighborhood_avg_info, annotation):
        n_avg, (h_, w_) = neighborhood_avg_info
        (h, w) = (np.ceil(h_ * self.downscale_ratio).astype(np.int), np.ceil(w_ * self.downscale_ratio).astype(np.int))
        #REMOVE print(downscaled_img.shape, w, h, downscaled_img.shape[1] - w, downscaled_img.shape[0] - h)
        (l, t, r, b), (l_, t_, r_, b_) = self.get_inner_outer_boundaries(downscaled_img.shape, annotation, 2)
        downscaled_img[t_:b_, l_:r_] = 0
        sw_an = np.zeros((downscaled_img.shape[0] - h, downscaled_img.shape[1] - w, 4 * 3))
        for i in range(downscaled_img.shape[0] - h):
            for j in range(downscaled_img.shape[1] - w):
                sw_an[i, j, :] = np.concatenate([
                    downscaled_img[i:i + h, j].mean(axis=0),
                    downscaled_img[i, j:j + w].mean(axis=0),
                    downscaled_img[i:i + h, j + w].mean(axis=0),
                    downscaled_img[i + h, j:j + w].mean(axis=0)
                ], axis=0)
        # print((dist_map - n_avg).mean(axis=2).shape)
        distmap = np.linalg.norm(sw_an - n_avg, axis=2)
        nms_img = non_max_suppression(distmap.max() - distmap, (5, 5), val=None)
        # imshow(distmap)
        # imshow(nms_img)
        points = list(zip(*np.where(nms_img != 0)))
        points.sort(key=lambda x: distmap[x[0], x[1]])
        # print(points)
        # print(max(list(map(lambda x: distmap[x[0], x[1]], points))))

        # print(n_avg)
        # print("distmap shape", distmap.shape)
        # index = np.unravel_index(distmap.argmin(), distmap.shape)
        # print("match point", index, distmap[index[0], index[1]])
        centers_info = [(((point[1] / distmap.shape[1]), (point[0] / distmap.shape[0])), distmap[point[0], point[1]])
                        for point in points[:self.max_aug * 3]]
        # print(sw_an[index[0], index[1]])
        return centers_info

    def iou(self, a1, a2):
        (l1, t1, r1, b1) = self.to_ltrb(a1)
        (l2, t2, r2, b2) = self.to_ltrb(a2)
        (l, t, r, b) = [max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2)]
        intersection = max(0, r - l) * max(0, b - t)
        union = (r1 - l1) * (b1 - t1) + (r2 - l2) * (b2 - l1) - intersection
        return intersection / union

    def vert_bound_dist_score(self, anno):
        return min(anno[2], 1 - anno[2]) / 0.5

    def area(self, anno):
        return anno[3] * anno[4]

    def find_k_best_shuffles(self, old_annotations, new_centers_info):
        drop_crossings_filter = lambda a: (
                (a[2] - a[4] / 2) > 0 and (a[2] + a[4] / 2) < 1 and (a[1] - a[3] / 2) > 0 and (
                a[1] + a[3] / 2) < 1)
        # scores = [self.area(anno) * self.vert_bound_dist_score(center)
        #           for anno, (center, err) in zip(old_annotations, new_centers_info)]
        new_annotations = [np.asarray([anno[0], center[0], center[1], anno[3], anno[4]]) for ((center, err), anno) in
                           zip(new_centers_info, old_annotations)]
        # new_annotations.sort(key=lambda x:- self.area(x) * self.vert_bound_dist_score(x))
        sorted_anno_w_pso = [(anno, pos) for anno, pos in sorted(zip(new_annotations, range(len(old_annotations))),
                                                                 key=lambda pair: - self.area(
                                                                     pair[0]) * self.vert_bound_dist_score(pair[0]))]
        selected_annotations = []
        old_position_mapping = []
        for anno_w_pos in sorted_anno_w_pso:
            anno = anno_w_pos[0]
            old_pos = anno_w_pos[1]
            if len(selected_annotations) < self.max_obj_shuffle:
                # print(type(old_annotations), type(selected_annotations))
                possible_intersection = max(
                    [self.iou(anno, old_anno) for old_anno in list(old_annotations) + selected_annotations])
                # print("possible_intersection", possible_intersection)
                # print("Yo", possible_intersection, 0.0, possible_intersection==0.0)
                if possible_intersection == 0.0 and self.area(anno) > 0.0003 and drop_crossings_filter(anno):
                    selected_annotations.append(anno)
                    old_position_mapping.append(old_pos)
        #REMOVE print(old_position_mapping)
        return selected_annotations, old_position_mapping

    def morph_image(self, image_orig, old_annotations, new_annotations, pos_map):
        def weight_to_out(x, y, l, t, r, b, l_, t_, r_, b_):
            crop = lambda z, p, q: p if z < p else (q if z > q else z)
            out_dist = min(min(x - l_, r_ - x), min(y - t_, b_ - y))
            x_ = crop(x, l, r)
            y_ = crop(y, t, b)
            in_dist = np.linalg.norm([x - x_, y - y_])
            return out_dist / (out_dist + in_dist)

        image = image_orig.copy()
        for old_anno, new_anno in zip(old_annotations[pos_map], new_annotations):
            (l, t, r, b), (l_, t_, r_, b_) = self.get_inner_outer_boundaries(image.shape, new_anno)
            (lo, to, ro, bo), (lo_, to_, ro_, bo_) = self.get_inner_outer_boundaries(image.shape, old_anno)
            image[t:b, l:r] = image[to:to + (b - t), lo:lo + (r - l)]
            frame_boxes = [
                (zip(range(t_, t), range(to_, to)), zip(range(l_, r_), range(lo_, ro_))),  # top frame
                (zip(range(b, b_), range(bo, bo_)), zip(range(l_, r_), range(lo_, ro_))),  # bottom frame
                (zip(range(t, b), range(to, bo)), zip(range(l_, l), range(lo_, lo))),  # middle part of left frame
                (zip(range(t, b), range(to, bo)), zip(range(r, r_), range(ro, ro_)))  # middle part of right frame
            ]
            for frame_box in frame_boxes:
                for i, io in frame_box[0]:
                    for j, jo in frame_box[1]:
                        w2o = weight_to_out(j, i, l, t, r, b, l_, t_, r_, b_)
                        image[i, j] = (image[i, j] * (1 - w2o) + image_orig[io, jo] * w2o).astype(np.uint8)

        # imshow([image])
        return image

    def augment(self, image, annotations):
        neighbourhood_avg_infos = self.get_all_annotations_neighbourhood_avg(image, annotations)
        downscaled_img = cv.resize(image, None, fx=self.downscale_ratio, fy=self.downscale_ratio,
                                   interpolation=cv.INTER_CUBIC)
        new_centers_info = [self.find_best_shuffle_pos(downscaled_img.copy(), n_avg_info, anno) for (n_avg_info, anno)
                            in
                            zip(neighbourhood_avg_infos, annotations)]

        new_annotations_set = []
        pos_maps = []
        # print(annotations)
        min_found_shuffle_matches = min([len(n_c_info) for n_c_info in new_centers_info], default=0)
        for i in range(min_found_shuffle_matches):
            new_annos, old_pmap = self.find_k_best_shuffles(annotations, [n_c_info[i] for n_c_info in new_centers_info])
            #REMOVE print(len(new_annos))
            if len(new_annos) > 0:
                new_annotations_set.append(new_annos)
                pos_maps.append(old_pmap)
            if len(new_annotations_set) >= self.max_aug:
                break

        return [(self.morph_image(image, annotations, new_annotations, pos_map),
                 (new_annotations + list(annotations))) for new_annotations, pos_map in
                zip(new_annotations_set, pos_maps)]
