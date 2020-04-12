import abc
from typing import Tuple, List, Any
import numpy as np


class SampleGeneration():
    def __init__(self, state):
        self.rnd = np.random.RandomState(state)

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> Tuple[List[Tuple[Any, float]], List[int]]:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


class SimpleMultichangeSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int,
        n_modes: int = 13,
        probs: List[float] = None,
        change_period: float = 1000,
        change_period_noise: float = 1,
        change_amount: float = 100,
        change_amount_noise: float = 0.1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        sample = []
        change_points = []
        change_ids = []
        mu = np.arange(n_modes, dtype=np.float64)
        limit = int(self.rnd.normal(change_period, change_period_noise))
        if probs is None:
            probs = 1 / np.arange(1, n_modes + 1)
            probs /= probs.sum()
        for i in range(size):
            if i == limit:
                limit += int(self.rnd.normal(change_period, change_period_noise))
                mu += self.rnd.normal(change_amount, change_amount_noise, size=n_modes)
                change_points.append(i)
                change_ids.append((0, -1))
            mode = self.rnd.choice(n_modes, p=probs)
            sample.append(((0, 0), self.rnd.normal(mu[mode], 1)))
        return sample, change_points, change_ids


class TerminalOrderChangeSampleGenerarion(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array([self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_terminal_id = self.rnd.choice(n_terminals)
                    change = []
                    for j in range(n_clients):
                        first_terminal_id = terminal_order[j].argmax()
                        terminal_order[i][hacked_terminal_id], terminal_order[i][first_terminal_id] = terminal_order[i][first_terminal_id], terminal_order[i][hacked_terminal_id]
                        change.append((hacked_terminal_id, first_terminal_id))

                    change_points.append(i)
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    for j, (hacked_terminal_id, first_terminal_id)  in enumerate(change):
                        terminal_order[i][hacked_terminal_id], terminal_order[i][first_terminal_id] = terminal_order[i][first_terminal_id], terminal_order[i][hacked_terminal_id]
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points


class ClientOrderChangeSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = self.rnd.choice(n_clients)
                    first_client_id = client_order.argmax()
                    client_order[hacked_client_id], client_order[first_client_id] = client_order[first_client_id], client_order[hacked_client_id]
                    change = (hacked_client_id, first_client_id)
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    hacked_client_id, first_client_id = change
                    client_order[hacked_client_id], client_order[first_client_id] = client_order[first_client_id], client_order[hacked_client_id]
                    change = None
                client_probabilities = 1 / (1 + client_order)
                client_probabilities /= client_probabilities.sum()

            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points


class ClientTerminalsReorderSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        print(n_clients)
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = self.rnd.choice(n_clients)
                    new_terminal_order = self.rnd.choice(n_terminals, size=n_terminals, replace=False)
                    change = (hacked_client_id, terminal_order[hacked_client_id])
                    terminal_order[hacked_client_id] = new_terminal_order
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    hacked_client_id, old_terminal_order = change
                    terminal_order[hacked_client_id] = old_terminal_order
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points


class ClientAverageAmountSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100,
        delta: float = 10,
        delta_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = self.rnd.choice(n_clients)
                    d = self.rnd.normal(delta, delta_noise)
                    mu[hacked_client_id] += d
                    change = (hacked_client_id, d)
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    hacked_client_id, d = change
                    mu[hacked_client_id] -= d
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points


class ChangeWithClientSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100,
        delta: float = 10,
        delta_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        a = self.rnd.lognormal(mean=np.log(10), size=n_clients)
        b = self.rnd.lognormal(mean=np.log(10), size=n_terminals)
        mu = a.reshape(-1, 1) @ b.reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = self.rnd.choice(n_clients)
                    d = b * self.rnd.lognormal(mean=np.log(10))
                    mu[hacked_client_id] = mu[hacked_client_id] + d
                    client_anomaly = client_order.copy() + 1
                    new_client_probabilities = 1 / (1 + client_anomaly)
                    new_client_probabilities[hacked_client_id] += 1.
                    new_client_probabilities /= new_client_probabilities.sum()

                    terminal_anomaly = self.rnd.choice(n_terminals, size=n_terminals, replace=False)
                    new_terminal_probabilities = 1 / (1 + terminal_order[hacked_client_id]) + 1 / (1 + terminal_anomaly)
                    new_terminal_probabilities /= new_terminal_probabilities.sum()

                    change = (hacked_client_id, d)

                    client_probabilities = new_client_probabilities
                    terminal_probabilities[hacked_client_id] = new_terminal_probabilities
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    hacked_client_id, d = change
                    mu[hacked_client_id] -= d

                    client_probabilities = 1 / (1 + client_order)
                    client_probabilities /= client_probabilities.sum()

                    terminal_probabilities[hacked_client_id] = 1 / (1 + terminal_probabilities[hacked_client_id])
                    terminal_probabilities[hacked_client_id] /= terminal_probabilities[hacked_client_id].sum()

                    change = None
            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points

class ChangeWithTerminalSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100,
        delta: float = 10,
        delta_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int]]:
        client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        a = self.rnd.lognormal(mean=np.log(10), size=n_clients)
        b = self.rnd.lognormal(mean=np.log(10), size=n_terminals)
        mu = a.reshape(-1, 1) @ b.reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_terminal_id = self.rnd.choice(n_terminals)
                    d = a * self.rnd.lognormal(mean=np.log(10))
                    mu[:, hacked_terminal_id] = mu[:, hacked_terminal_id] + d

                    change = (hacked_terminal_id, d)
                else:
                    limit += int(self.rnd.normal(change_period, change_period_noise))
                    hacked_terminal_id, d = change
                    mu[:, hacked_terminal_id] -= d

                    change = None
            client_id = self.rnd.choice(n_clients, p=client_probabilities)
            terminal_id = self.rnd.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points


class ChangeSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_account1s: int = 20,
        n_account2s: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100,
        delta: float = 10,
        delta_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        #client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = np.ones(n_account1s) / n_account1s #1 / (1 + client_order)
        #client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [self.rnd.choice(n_account2s, size=n_account2s, replace=False) for _ in range(n_account1s)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        change_ids = []
        a = np.ones(n_account1s, dtype=float) * 10 #self.rnd.lognormal(mean=np.log(10), size=n_clients)
        b = np.ones(n_account2s, dtype=float) * 10 #self.rnd.lognormal(mean=np.log(10), size=n_terminals)
        mu = a.reshape(-1, 1) @ b.reshape(1, -1)
        limit = int(self.rnd.normal(change_period, change_period_noise))

        change = None
        is_client_change = False
        for i in range(size):
            if i == limit:
                if change is None:
                    is_client_change = not is_client_change
                    if is_client_change:
                        change_points.append(i)
                        limit += change_interval
                        hacked_client_id = self.rnd.choice(n_account1s)
                        d = b * 10 # self.rnd.lognormal(mean=np.log(10))
                        mu[hacked_client_id] = mu[hacked_client_id] + d
                        #client_anomaly = client_order.copy() + 1
                        new_client_probabilities = np.ones(n_account1s) #1 / (1 + client_anomaly)
                        new_client_probabilities[hacked_client_id] += 1.
                        new_client_probabilities /= new_client_probabilities.sum()

                        #terminal_anomaly = self.rnd.choice(n_terminals, size=n_terminals, replace=False)
                        #new_terminal_probabilities = 1 / (1 + terminal_order[hacked_client_id]) + 1 / (1 + terminal_anomaly)
                        #new_terminal_probabilities /= new_terminal_probabilities.sum()

                        change = (hacked_client_id, d, True)
                        change_ids.append((hacked_client_id, -1))

                        client_probabilities = new_client_probabilities
                        #terminal_probabilities[hacked_client_id] = new_terminal_probabilities
                    else:
                        change_points.append(i)
                        limit += change_interval
                        hacked_terminal_id = self.rnd.choice(n_account2s)
                        d = a * 10 #self.rnd.lognormal(mean=np.log(10))
                        mu[:, hacked_terminal_id] = mu[:, hacked_terminal_id] + d

                        change = (hacked_terminal_id, d, False)
                        change_ids.append((-1, hacked_terminal_id))
                else:
                    change_points.append(i)
                    hacked_id, d, is_client_change = change
                    if is_client_change:
                        change_ids.append((hacked_id, -1))
                        limit += int(self.rnd.normal(change_period, change_period_noise))
                        mu[hacked_id] -= d

                        client_probabilities = np.ones(n_account1s) / n_account1s #1 / (1 + client_order)
                        #client_probabilities /= client_probabilities.sum()

                        #terminal_probabilities[hacked_id] = 1 / (1 + terminal_probabilities[hacked_id])
                        #terminal_probabilities[hacked_id] /= terminal_probabilities[hacked_id].sum()
                    else:
                        change_ids.append((-1, hacked_id))
                        limit += int(self.rnd.normal(change_period, change_period_noise))
                        mu[:, hacked_id] -= d
                    change = None
            client_id = self.rnd.choice(n_account1s, p=client_probabilities)
            terminal_id = self.rnd.choice(n_account1s, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), self.rnd.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points, change_ids


class StillSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int = 100000,
        n_clients: int = 20,
        n_terminals: int = 20,
        change_period: int = 1000,
        change_period_noise: float = 1,
        change_interval: int = 100,
        delta: float = 10,
        delta_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        #client_order = self.rnd.choice(n_clients, size=n_clients, replace=False)
        #client_probabilities = 1 / (1 + client_order)
        #client_probabilities /= client_probabilities.sum()
        #terminal_order = np.array(
        #    [self.rnd.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        #terminal_probabilities = 1 / (1 + terminal_order)
        #terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        #mu = 5
        for i in range(size):
            sample.append(((0, -1), self.rnd.uniform(0, 1)))
        return sample, [], []


class LogExpChangeSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int,
        probs: List[float] = None,
        change_period: float = 200,
        change_period_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        sample = []
        change_points = []
        change_ids = []
        mu = 0
        limit = int(self.rnd.normal(change_period, change_period_noise))
        is_normal = True
        for i in range(size):
            if i == limit:
                limit += int(self.rnd.normal(change_period, change_period_noise))
                change_points.append(i)
                change_ids.append((0, -1))
                is_normal = not is_normal
            if is_normal:
                sample.append(((0, 0), self.rnd.normal(loc=mu, scale=1)))
            else:
                sample.append(((0, 0), self.rnd.exponential(scale=1)))
        return sample, change_points, change_ids


class OriginalExperiment1UniformSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int,
        probs: List[float] = None,
        change_period: float = 200,
        change_period_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        sample = []
        change_points = []
        change_ids = []
        param = 5
        param_noise = 1
        limit = int(self.rnd.normal(change_period, change_period_noise))
        for i in range(size):
            if i == limit:
                limit += int(self.rnd.normal(change_period, change_period_noise))
                param = abs(param + self.rnd.uniform(-param_noise, param_noise))
                change_points.append(i)
                change_ids.append((0, -1))
            sample.append(((0, -1), self.rnd.uniform(-param, param)))

        return sample, change_points, change_ids

class OriginalExperiment2NormalUniformMixtureSampleGeneration(SampleGeneration):
    def generate(
        self,
        size: int,
        probs: List[float] = None,
        change_period: float = 200,
        change_period_noise: float = 1
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[Tuple[int, int]]]:
        sample = []
        change_points = []
        change_ids = []
        param = 0.9
        param_noise = 0.05
        limit = int(self.rnd.normal(change_period, change_period_noise))
        for i in range(size):
            if i == limit:
                limit += int(self.rnd.normal(change_period, change_period_noise))
                param = param + self.rnd.uniform(-param_noise, param_noise)
                if param > 1:
                    param = 1 - (param - 1)
                change_points.append(i)
                change_ids.append((0, -1))
            if self.rnd.choice(2, p=[param, 1 - param]) == 1:
                sample.append(((0, -1), self.rnd.uniform(-7, 7)))
            else:
                sample.append(((0, -1), self.rnd.normal(loc=0, scale=1)))
        return sample, change_points, change_ids