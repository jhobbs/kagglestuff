import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data():
    with open('train.csv') as training_data:
        rows = training_data.readlines()[1:]
    
    ys = []
    xs = []
    for row in rows:
        raw_numbers = list(map(int, row.split(',')))
        ys.append(raw_numbers[0])
        xs.append(torch.tensor(raw_numbers[1:]).float() / 255.0 )
    return torch.stack(xs), torch.tensor(ys)


def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(784,10)
    )

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    xs, ys = get_data()

    N = xs.size(0)
    test_ratio = 0.2

    # 1. Shuffle indices
    perm = torch.randperm(N)

    # 2. Compute split point
    test_size = int(N * test_ratio)

    # 3. Take slices
    test_idx  = perm[:test_size]
    train_idx = perm[test_size:]

    xs_train = xs[train_idx]
    ys_train = ys[train_idx]
    xs_test  = xs[test_idx]
    ys_test  = ys[test_idx]


    train_ds = TensorDataset(xs_train, ys_train)

    train_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
    )

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (xb, yb) in enumerate(train_loader):
            logits = model(xb)
            loss = loss_function(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    print(loss.item())

        model.eval()
        with torch.no_grad():
            logits = model(xs_test)
            loss = loss_function(logits, ys_test)
            preds = logits.argmax(dim=1)
            acc = (preds == ys_test).float().mean().item()
    
        print("test loss:", loss.item())
        print("test accuracy:", acc)


if __name__ == "__main__":
    main()
