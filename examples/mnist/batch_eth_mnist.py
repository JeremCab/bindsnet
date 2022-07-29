import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from functools import partialmethod

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015, TenClasses,DiehlAndCook2015v3, SCNN
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network.topology import Connection, Conv2dConnection

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=12)
parser.add_argument("--n_classes", type=int, default=15)
parser.add_argument("--norm", type=float, default=50)
parser.add_argument("--metric_norm", type=float, default=2)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--cpu", dest="gpu", action="store_false")
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
n_updates = args.n_updates
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
train = args.train
plot = args.plot
gpu = args.gpu
n_classes= args.n_classes
metric_norm = args.metric_norm
norm = args.norm

update_steps = int(n_train / batch_size / n_updates)
update_interval = update_steps * batch_size

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity


class SnnMnist:
    def __init__(self, gpu, seed, n_workers, **kwargs):
        if gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            self.device = "cuda"
        else:
            torch.manual_seed(seed)
            self.device = "cpu"
            gpu = False
        self.gpu = gpu
        torch.set_num_threads(os.cpu_count() - 1)
        print("Running on Device = ", self.device)

        self.n_workers = n_workers if n_workers != -1 else 0  # gpu * 1 * torch.cuda.device_count()

        self.ims = {}
        self._build_network(**kwargs)

    def _build_network(self, n_neurons, exc, inh, dt, theta_plus, n_classes, norm=5, **kwargs):
#         self.network = DiehlAndCook2015v3(
#             n_inpt=784,
#             n_neurons=n_neurons,
#             inh=inh,
#             dt=dt,
#             norm=norm,  # 78.4,
#             nu=(1e-4, 1e-2),
#             theta_plus=theta_plus,
#             inpt_shape=(1, 28, 28),
#             **kwargs
#         )
#         self.network = TenClasses(
#             n_input=784,
#             n_neurons=n_neurons,
#             n_classes=n_classes,
#             inter_inh=0,
#             exc=exc,
#             inh=inh,
#             dt=dt,
#             norm=norm,  # 78.4,
#             nu=(1e-4, 1e-2),
#             theta_plus=theta_plus,
#             input_shape=(1, 28, 28),
#             **kwargs
#         )
        self.network = SCNN(
            n_input=784,
            n_neurons=n_neurons,
            n_classes=n_classes,
            inter_inh=2,
            exc=exc,
            inh=inh,
            dt=dt,
            norm=norm,  # 78.4,
            nu=(1e-4, 1e-2),
            theta_plus=theta_plus,
            input_shape=(1, 28, 28),
            **kwargs
        )

        if self.gpu:
            self.network.to("cuda")

        # Neuron assignments and spike proportions.
        self.n_classes = 10
        self.assignments = -torch.ones(n_neurons, device=self.device)
        self.proportions = torch.zeros((n_neurons, self.n_classes), device=self.device)
        self.rates = torch.zeros((n_neurons, self.n_classes), device=self.device)

        # Sequence of accuracy estimates.
        self.accuracy = {"all": [], "proportion": []}

        # Set up monitors for spikes
        for layer in self.network.layers:
            if layer in ("X","Y","11","IF11"):
                monitor = Monitor(self.network.layers[layer], state_vars=["s"] if layer!= "11" else ["s","v"], time=int(time / dt), device=self.device)
                self.network.add_monitor(monitor, name="%s_monitor" % layer)

        self.labels = torch.zeros(update_interval, device=self.device)
        self.spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=self.device)
        
    def load_dataset(self, time, dt, train):
        # Load MNIST data.
        self.dataset = MNIST(
            PoissonEncoder(time=time, dt=dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "MNIST"),
            download=True,
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
            ),
        )

    def create_dataloader(self):
        # Create a dataloader to iterate and batch data
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=self.gpu,
            drop_last=True
        )

    def process_batch(self):
        # Get next input sample.
        self.inputs = {"X": batch["encoded_image"]}
        if self.gpu:
            self.inputs = {k: v.cuda() for k, v in self.inputs.items()}

        # Run the network on the input.
        self.network.run(inputs=self.inputs, time=time, input_time_dim=1)

        # Record spikes and labels.
        i = step * batch_size % update_interval
        self.labels[i: i + batch_size] = batch["label"]
        s = self.network.monitors["Y_monitor"].get("s").permute((1, 0, 2))
        self.spike_record[i: i + batch_size] = s

    def set_normalize(norm):
        def function(self, norm=1):
            if self.norm is not None:
                w_abs_sum = self.w.pow(norm).sum(0).pow(1 / norm).unsqueeze(0)
                w_abs_sum[w_abs_sum == 0] = 1.0
                self.w *= self.norm / w_abs_sum

        Connection.normalize = partialmethod(function, norm=norm)

    def save_plots(self, path="plots"):
        image = batch["image"][:, 0].view(28, 28)
        inpt = self.inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
        lable = batch["label"][0]


#         square_weights = get_square_weights(
#             self.network.connections[("X", "Y")].w.view(784, n_neurons), n_sqrt, 28
#         )
#         self.weights_im = plot_weights(square_weights, im=self.weights_im, wmin = square_weights.min(), wmax= square_weights.max()+0.5)
        
        square_weights = get_square_weights(
            self.network.connections[("IF11", "11")].w.view(49, 7**2), 7, 7
        )
        self.ims["weights"] = plot_weights(square_weights, im=self.ims.get("weights"), wmin = square_weights.min(), wmax= square_weights.max()+0.5)
        
        identity_weights = self.network.connections[("X", "IF11")].w
        self.ims["identity_weights"] = plot_weights(identity_weights, im=self.ims.get("identity_weights"), wmin = identity_weights.min(), wmax= identity_weights.max()+0.5)
        
        
        rec_weights = self.network.connections[("Y", "Y")].w
        self.ims["rec_weights"] = plot_weights(rec_weights, im=self.ims.get("rec_weights"), wmin = rec_weights.min(), wmax = rec_weights.max())

        square_assignments = get_square_assignments(self.assignments, n_sqrt)
        self.ims["assigns"] = plot_assignments(square_assignments, im=self.ims.get("assigns"))
        
        spikes_ = {
            monitor.obj: monitor.get("s")[:, 0].contiguous() for monitor in
            self.network.monitors.values()
        }
        self.ims["spikes"], self.ims["spikes_axes"] = plot_spikes(spikes_, ims=self.ims.get("spikes"), axes=self.ims.get("spikes_axes"))

        voltages_ = {
            self.network.monitors["11_monitor"].obj: self.network.monitors["11_monitor"].get("v")#[:, 0].contiguous()
        }
        self.ims["voltages"], self.ims["voltages_axes"] = plot_voltages(voltages_, ims=self.ims.get("voltages"), axes=self.ims.get("voltages_axes"), plot_type="line")
        
        self.ims["inpt_axes"], self.ims["inpt"] = plot_input(
            image, inpt, label=lable, axes=self.ims.get("inpt_axes"), ims=self.ims.get("inpt")
        )
        
        
        self.ims["perf_ax"] = plot_performance(
            self.accuracy, x_scale=update_steps * batch_size, ax= self.ims.get("perf_ax")
        )

        for n in plt.get_fignums():
            os.makedirs(path, exist_ok=True)
            plt.figure(n)
            plt.savefig(f"{path}/plot_{epoch}_{step // update_steps :02d}_{n}", bbox_inches="tight")

        with open(f"{path}/plot_{epoch}_{step // update_steps :02d}.txt", "w") as file:
            for k in self.accuracy:
                file.write(k + "\n" + " ".join(map("{:.2f}".format, self.accuracy[k])) + "\n")
        file.close()

    def predict(self):
        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(self.labels, device=self.device)

        # Get network predictions.
        all_activity_pred = all_activity(
            spikes=self.spike_record, assignments=self.assignments, n_labels=self.n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=self.spike_record,
            assignments=self.assignments,
            proportions=self.proportions,
            n_labels=self.n_classes,
        )

        # Compute network accuracy according to available classification strategies.
        self.accuracy["all"].append(
            100
            * torch.sum(label_tensor.long() == all_activity_pred).item()
            / len(label_tensor)
        )
        self.accuracy["proportion"].append(
            100
            * torch.sum(label_tensor.long() == proportion_pred).item()
            / len(label_tensor)
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (
                self.accuracy["all"][-1],
                np.mean(self.accuracy["all"]),
                np.max(self.accuracy["all"]),
            )
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
            " (best)\n"
            % (
                self.accuracy["proportion"][-1],
                np.mean(self.accuracy["proportion"]),
                np.max(self.accuracy["proportion"]),
            )
        )

    def assign_labels(self):
        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(self.labels, device=self.device)

        # Assign labels to excitatory layer neurons.
        self.assignments, self.proportions, self.rates = assign_labels(
            spikes=self.spike_record,
            labels=label_tensor,
            n_labels=self.n_classes,
            rates=self.rates,
        )

snn_mnist = SnnMnist(**vars(args))
SnnMnist.set_normalize(norm=metric_norm)
snn_mnist.load_dataset(time=time, dt=dt, train=True)
# Train the network.
print("\nBegin training...")
start = t()
for epoch in range(n_epochs):
    snn_mnist.create_dataloader()

    print("\nProgress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
    start = t()

    pbar_training = tqdm(total=n_train)
    for step, batch in enumerate(snn_mnist.dataloader):
        if step * batch_size > n_train:
            break
<<<<<<< HEAD
#         if epoch == 0 and step == update_steps:
#             snn_mnist.network.connections[("Y", "Y")].w[:]=snn_mnist.network.inhib_weights(inh=inh, n_classes=n_classes, inter_inh=3)

        # Assign labels to excitatory neurons.
        if step % update_steps == 0 and step > 0:
            snn_mnist.predict()
            snn_mnist.assign_labels()
            snn_mnist.labels = []
        # Process batch
        snn_mnist.process_batch()
=======

        # Assign labels to excitatory neurons.
        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Remember labels.
        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
>>>>>>> 206e6c5bda47b1042b55a347c5d67593126d67d9

        # Optionally plot various simulation information.
        if plot and step % update_steps == 0:
            snn_mnist.save_plots(f"plot_norm{metric_norm}_mean{norm}_train")

        # Reset state variables.
        snn_mnist.network.reset_state_variables()
        pbar_training.update(batch_size)
    pbar_training.close()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTraining complete.\n")

# Test the network.

snn_mnist.load_dataset(time=time, dt=dt, train=False)
snn_mnist.create_dataloader()

print("\nBegin testing...\n")
snn_mnist.network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")

for step, batch in enumerate(snn_mnist.dataloader):
    if step * batch_size > n_test:
        break

    snn_mnist.process_batch()

    if step % update_steps == 0 and step > 0:
        snn_mnist.predict()
        snn_mnist.save_plots(f"plot_norm{metric_norm}_mean{norm}_test")

    snn_mnist.network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

print("\nTesting complete.\n")
plt.close('all')
del snn_mnist
