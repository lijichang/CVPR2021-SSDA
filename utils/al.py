
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
def sample_selection(args, G, F1, target_loader_unl, known_labels, use_gpu):
    G.eval()
    F1.eval()
    # p = look in original paper for clues
    # alpha = look in original paper for clues
    s = False

    for data, target, idx in target_loader_unl:
        data, target, idx = data.type(torch.FloatTensor), target.type(torch.LongTensor), idx.type(
            torch.LongTensor).cpu().numpy()
        idx_filter = np.logical_not(np.isin(idx, known_labels))
        if (idx_filter == False).all():
            continue
        idx_filter = torch.from_numpy(idx_filter.astype(np.uint8))
        data, target = data[idx_filter], target[idx_filter]

        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            feat = G(data)
            out1 = F1(feat)

        ## selection strategy
        if (args.strategy == "Margin_Sampling") or (args.strategy == "Least_Confidence") or (args.strategy == "Entropy_Sampling") :
            if s is False:
                prob = F.softmax(out1, dim=1)
                s = prob.cpu()
                idxs = idx
                lbls = target.cpu()
            else:
                prob = F.softmax(out1, dim=1)
                s = torch.cat((s, prob.cpu()))
                idxs = np.concatenate((idxs, idx))
                lbls = torch.cat((lbls, target.cpu()))
        elif args.strategy == "Random_Sampling":
            if s is False:
                s = 1.0
                idxs = idx
                lbls = target.cpu()
            else:
                s = 1.0
                idxs = np.concatenate((idxs, idx))
                lbls = torch.cat((lbls, target.cpu()))
        elif args.strategy == "Kmeans_Sampling":
            if s is False:
                s = feat.cpu()  # only collect feature (output of G)
                idxs = idx
                lbls = target.cpu()
            else:
                s = torch.cat((s, feat.cpu())) # only collect feature (output of G)
                idxs = np.concatenate((idxs, idx))
                lbls = torch.cat((lbls, target.cpu()))
        elif args.strategy == "None":
            pass
        else:
            raise ValueError('Strategy cannot be recognized.')

    if args.strategy == "Margin_Sampling":
        idxs = idxs
        labels = lbls.cpu().numpy()
        scores = s.cpu().numpy()
        sorted = np.sort(scores, axis=1)
        U = sorted[:, 0] - sorted[:, 1]
        #new_labels = U.sort()[1][:args.b]
        new_labels = np.argsort(U)[:args.b]
        gt = labels[new_labels].tolist()

        new_labels = idxs[new_labels]
        new_labels = new_labels.tolist()
    elif args.strategy == "Least_Confidence":
        idxs = idxs
        labels = lbls.cpu().numpy()
        scores = s.cpu().numpy()
        U = np.max(scores, axis=1)
        #new_labels = U.sort()[1][:args.b]
        new_labels = np.argsort(U)[:args.b]
        gt = labels[new_labels].tolist()

        new_labels = idxs[new_labels]
        new_labels = new_labels.tolist()
    elif args.strategy == "Random_Sampling":

        idxs = idxs
        labels = lbls.cpu().numpy()
        rand_ix = np.arange(len(labels))
        # print("rand_ix", rand_ix)
        # print("labels", labels.shape)
        # print("rand_ix", rand_ix.shape)
        # print("rand_ix max", np.max(rand_ix))
        np.random.shuffle(rand_ix)
        new_labels = rand_ix[:args.b]
        # print("new_labels", new_labels)
        # print("new_labels", new_labels.shape)
        # print("new_labels max", np.max(new_labels))
        gt = labels[new_labels].tolist()
        new_labels = idxs[new_labels]
        new_labels = new_labels.tolist()
    elif args.strategy == "Kmeans_Sampling":
        idxs = idxs
        labels = lbls.cpu().numpy()
        s = s.numpy()
        cluster_learner = KMeans(n_clusters=args.b)
        cluster_learner.fit(s)

        cluster_idxs = cluster_learner.predict(s)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (s - centers) ** 2
        dis = dis.sum(axis=1)
        new_labels = np.array([np.arange(s.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(args.b)])
        gt = labels[new_labels].tolist()
        new_labels = idxs[new_labels]
        new_labels = new_labels.tolist()
    elif args.strategy == "Entropy_Sampling":
        idxs = idxs
        labels = lbls.cpu().numpy()
        scores = s.cpu().numpy()  # probablity
        log_scores = np.log(scores)
        U = (scores * log_scores).sum(axis=1)
        new_labels = np.argsort(U)[:args.b]
        gt = labels[new_labels].tolist()
        new_labels = idxs[new_labels]
        new_labels = new_labels.tolist()

    elif args.strategy == "NONE":
        new_labels = []
        gt = []

    else:
        raise ValueError('Strategy cannot be recognized.')
    return new_labels, gt