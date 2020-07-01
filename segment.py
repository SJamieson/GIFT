import argparse

from train.evaluation import EvaluationWrapper, CorrespondenceEstimator
from skimage.io import imread, imshow
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

parser=argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='train')
# parser.add_argument('--cfg', type=str, default='configs/GIFT-stage1.yaml')
parser.add_argument('--det_cfg', type=str, default='configs/eval/superpoint_det.yaml')
parser.add_argument('--desc_cfg', type=str, default='configs/eval/gift_pretrain_desc.yaml')
parser.add_argument('--n_topics', type=int, default=8)
parser.add_argument("image_file", type=str)
# parser.add_argument('--match_cfg',type=str,default='configs/eval/match_v2.yaml')
flags=parser.parse_args()

if __name__=="__main__":
    det_name= flags.det_cfg
    desc_name= flags.desc_cfg
    # match_name=EvaluationWrapper.get_stem(flags.match_cfg)
    # correspondence_estimator=CorrespondenceEstimator(
    det_cfg = EvaluationWrapper.load_cfg(det_name)
    desc_cfg = EvaluationWrapper.load_cfg(desc_name)
    # self.load_cfg(match_cfg_file))

    # evaluator = EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
    detector = CorrespondenceEstimator.name2det[det_cfg['type']](det_cfg)
    descriptor = CorrespondenceEstimator.name2desc[desc_cfg['type']](desc_cfg)
    img = imread(flags.image_file)
    # print(img.shape)
    # imshow(img)
    # plt.show()

    kps, desc = detector(img)
    if desc_cfg['type']=='none':
        assert(desc is not None)
    else:
        desc = descriptor(img, kps)
    # print(kps)
    # print(desc)

    cluster = KMeans(n_clusters=flags.n_topics)
    labels: np.ndarray = cluster.fit_predict(desc, None)
    palette = sns.hls_palette(cluster.n_clusters)
    # print(labels)
    # print(labels.shape)
    colors = [(np.array(palette[l]) * 255).astype(np.int) for l in labels]
    for pt, color in zip(kps, colors):
        # print(pt)
        for dx in np.arange(-4, 4):
            for dy in np.arange(-4, 4):
                # if not (dx == 0 or dy == 0):
                #     continue
                img[pt[1] + 4 + dx, pt[0] + 4 + dy, :] = color
    imshow(img)
    plt.show()

