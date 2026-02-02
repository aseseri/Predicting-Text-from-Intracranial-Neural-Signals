import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder, TransformerDecoder
from .dataset import SpeechDataset
from .augmentations import GaussianSmoothing, TimeMasking, FeatureMasking


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    # --- Experiment 3 ---
    # model = TransformerDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["Transformer-hidden-nLayers"],
    #     layer_dim=args["Transformer-nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["Transformer-dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    # ).to(device)
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)
    # -----

    # Adding to resume training from a checkpoint: checks if a specific checkpoint path was provided in args
    if "checkpointPath" in args and args["checkpointPath"] and os.path.exists(args["checkpointPath"]):
        print(f"Resuming from: {args['checkpointPath']}")
        model.load_state_dict(torch.load(args["checkpointPath"]))
    # -------

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # --- Experiment 1 ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.01, # 0.1 to 0.01 for finer convergence
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args["lrStart"],
        total_steps=args["nBatch"],
        pct_start=0.3, # 30% of time warming up
        anneal_strategy='cos',
    )
    # ----------------------------------------

    # --- Baseline ---
    # loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args["lrStart"],
    #     betas=(0.9, 0.999),
    #     eps=0.1,
    #     weight_decay=args["l2_decay"],
    # )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=args["lrEnd"] / args["lrStart"],
    #     total_iters=args["nBatch"],
    # )
    # ---

    # --- Experiment 2 ---
    time_masker = TimeMasking(mask_prob=0.2, max_mask_len=30).to(device)
    # ----
    # --- Experiment 4 ---
    # time_masker = TimeMasking(mask_prob=0.4, max_mask_len=40).to(device)
    # ----
    # --- Experiment 7 ---
    # feat_masker = FeatureMasking(mask_prob=0.2, max_mask_len=20).to(device) 
    # ----

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # --- Experiment 9 ---
        # X = torch.log1p(X - X.min() + 1e-6)
        # ----

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
        # ---  Experiment 2 ---
        X = time_masker(X)
        # ------------------------------
        # --- Experiment 7 ---
        # X = feat_masker(X)
        # -----

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )
                    # --- Experiment 9 ---
                    # X = torch.log1p(X - X.min() + 1e-6)
                    # ----

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)
    # --- Experiment 3 ---
    # model = TransformerDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["Transformer-hidden-nLayers"],
    #     layer_dim=args["Transformer-nLayers"],
    #     nDays=nInputLayers,
    #     dropout=args["Transformer-dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    # ).to(device)
    # -------

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
