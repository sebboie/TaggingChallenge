"""Utility functions for training and testing."""
import os
import torch
import pandas as pd
from global_vars import *


def test_network(testing_data, net, loss_fn, steps):
    """Evaluate a network on the testset."""
    print("Evaluating on test data.")
    correct = torch.zeros(NO_CLASSES)
    total = torch.zeros(NO_CLASSES)
    test_loss = 0
    correct_sample_class = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(testing_data):
            images, labels = batch
            y_hat = net(images)
            test_loss += loss_fn(y_hat, labels).item()
            correct_batch = torch.ones_like(labels) - \
                torch.round(torch.abs(y_hat - labels))
            correct += torch.sum(correct_batch, dim=0)
            total += torch.sum(torch.ones_like(labels), dim=0)

            correct_sample_class += ((correct_batch.sum(dim=1) ==
                                     NO_CLASSES).sum())
            total_samples += labels.shape[0]

            if i % steps == steps-1:
                classified_correct = (correct / total)
                classified_correct = pd.DataFrame([classified_correct.numpy()],
                                                  columns=FOOD_CATEGORIES)

                print("Correctly classified tags in percent per category:")
                for col in classified_correct.columns:
                    print("{}: {:.2f}".format(
                        col, 100*classified_correct[col].iloc[0]))

                print("\n Correctly classified tags across categories: {}"
                      "".format(100 * classified_correct.sum(axis=1) /
                                len(classified_correct.columns)))

                print("\n Fully correct samples: {} out of {}"
                      "".format(correct_sample_class, total_samples))

                return test_loss/(steps*images.size()[0])


def train_network(training_data, net, loss_fn, optimizer, steps=1):
    """Train a network for `steps` steps."""
    if not os.path.exists("chkpts"):
        os.makedirs("chkpts")

    chkpt_files = sorted(os.listdir("chkpts"))
    epoch = 0
    if chkpt_files:
        print("Found a checkpoint file. Will resume it.")
        chkpt_file = chkpt_files[-1]
        checkpoint = torch.load(os.path.join("chkpts", chkpt_file))
        epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Training for {} steps.".format(steps))
    train_loss = 0.0
    for i, batch in enumerate(training_data):
        images, labels = batch
        optimizer.zero_grad()

        y_hat = net(images)
        loss = loss_fn(y_hat, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % steps == steps-1:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                os.path.join("chkpts", "chkpt_{:06d}.pt".format(epoch+1))
            )

            return epoch + steps, train_loss/(steps*images.size()[0])
