import abc
from typing import Tuple, List, Any
import numpy as np


class SampleGeneration():
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
            tau: float = 1000,
            tau_noise: float = 1,
            delta: float = 10,
            delta_noise: float = 0.1
    ) -> Tuple[List[Tuple[int, float]], List[int]]:
        sample = []
        change_points = []
        mu = np.arange(n_modes, dtype=np.float64)
        limit = int(np.random.normal(tau, tau_noise))
        if probs is None:
            probs = 1 / np.arange(1, n_modes + 1)
            probs /= probs.sum()
        for i in range(size):
            if i == limit:
                limit += int(np.random.normal(tau, tau_noise))
                mu += np.random.normal(delta, delta_noise, size=n_modes)
                change_points.append(i)
            mode = np.random.choice(n_modes, p=probs)
            sample.append((mode, np.random.normal(mu[mode], 1)))
        return sample, change_points


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
        client_order = np.random.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array([np.random.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(np.random.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_terminal_id = np.random.choice(n_terminals)
                    change = []
                    for j in range(n_clients):
                        first_terminal_id = terminal_order[j].argmax()
                        terminal_order[i][hacked_terminal_id], terminal_order[i][first_terminal_id] = terminal_order[i][first_terminal_id], terminal_order[i][hacked_terminal_id]
                        change.append((hacked_terminal_id, first_terminal_id))

                    change_points.append(i)
                else:
                    limit += int(np.random.normal(change_period, change_period_noise))
                    for j, (hacked_terminal_id, first_terminal_id)  in enumerate(change):
                        terminal_order[i][hacked_terminal_id], terminal_order[i][first_terminal_id] = terminal_order[i][first_terminal_id], terminal_order[i][hacked_terminal_id]
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = np.random.choice(n_clients, p=client_probabilities)
            terminal_id = np.random.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), np.random.normal(mu[client_id][terminal_id], 1)))
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
        client_order = np.random.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [np.random.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(np.random.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = np.random.choice(n_clients)
                    first_client_id = client_order.argmax()
                    client_order[hacked_client_id], client_order[first_client_id] = client_order[first_client_id], client_order[hacked_client_id]
                    change = (hacked_client_id, first_client_id)
                else:
                    limit += int(np.random.normal(change_period, change_period_noise))
                    hacked_client_id, first_client_id = change
                    client_order[hacked_client_id], client_order[first_client_id] = client_order[first_client_id], client_order[hacked_client_id]
                    change = None
                client_probabilities = 1 / (1 + client_order)
                client_probabilities /= client_probabilities.sum()

            client_id = np.random.choice(n_clients, p=client_probabilities)
            terminal_id = np.random.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), np.random.normal(mu[client_id][terminal_id], 1)))
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
        client_order = np.random.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [np.random.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(np.random.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = np.random.choice(n_clients)
                    new_terminal_order = np.random.choice(n_terminals, size=n_terminals, replace=False)
                    change = (hacked_client_id, terminal_order[hacked_client_id])
                    terminal_order[hacked_client_id] = new_terminal_order
                else:
                    limit += int(np.random.normal(change_period, change_period_noise))
                    hacked_client_id, old_terminal_order = change
                    terminal_order[hacked_client_id] = old_terminal_order
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = np.random.choice(n_clients, p=client_probabilities)
            terminal_id = np.random.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), np.random.normal(mu[client_id][terminal_id], 1)))
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
        client_order = np.random.choice(n_clients, size=n_clients, replace=False)
        client_probabilities = 1 / (1 + client_order)
        client_probabilities /= client_probabilities.sum()
        terminal_order = np.array(
            [np.random.choice(n_terminals, size=n_terminals, replace=False) for _ in range(n_clients)])
        terminal_probabilities = 1 / (1 + terminal_order)
        terminal_probabilities /= terminal_probabilities.sum(axis=-1)

        sample = []
        change_points = []
        mu = np.arange(n_terminals, dtype=np.float64).reshape(-1, 1) @ np.arange(n_clients, dtype=np.float64).reshape(1, -1)
        limit = int(np.random.normal(change_period, change_period_noise))

        change = None
        for i in range(size):
            if i == limit:
                if change is None:
                    change_points.append(i)
                    limit += change_interval
                    hacked_client_id = np.random.choice(n_clients)
                    d = np.random.normal(delta, delta_noise)
                    mu[hacked_client_id] += d
                    change = (hacked_client_id, d)
                else:
                    limit += int(np.random.normal(change_period, change_period_noise))
                    hacked_client_id, d = change
                    mu[hacked_client_id] -= d
                    change = None
                terminal_probabilities = 1 / (1 + terminal_order)
                terminal_probabilities /= terminal_probabilities.sum(axis=-1)

            client_id = np.random.choice(n_clients, p=client_probabilities)
            terminal_id = np.random.choice(n_clients, p=terminal_probabilities[client_id])
            sample.append(((client_id, terminal_id), np.random.normal(mu[client_id][terminal_id], 1)))
        return sample, change_points