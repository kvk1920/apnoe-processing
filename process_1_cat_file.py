import os
import sys

import typing as tp

import numpy as np
import scipy.stats as sps


APNOE_LEFT_DURATION: float = 5.0  # левая половинка апноэ
APNOE_RIGHT_DURATION: float = 10.0  # правая половинка апное
STEP_DURATION = 0.01  # шаг для подсчёта частоты
BOOTSTRAP_PROHIBITED_TIME = 1.0  # ближе такого времени незя брать элементы бутстрепной выборки
BOOTSTRAP_SIZE = 1000  # размер бутстрепной выборки

# эти константы лучше не трогать, они высчитываются через те, что выше
APNOE_DURATION = APNOE_LEFT_DURATION + APNOE_RIGHT_DURATION
PLOT_SIZE = int(round((APNOE_DURATION - 1.0) / STEP_DURATION))


class Statistic:
    def __init__(self) -> None:
        self.min = int(1e9)
        self.max = int(-1e9)
        self.min_time = 0.0
        self.max_time = 0.0
        self.pvalue = None

    def update(self, cur_value: int, cur_time: float) -> None:
        if cur_value < self.min:
            self.min = cur_value
            self.min_time = cur_time
        if cur_value > self.max:
            self.max = cur_value
            self.max_time = cur_time

    @property
    def dif(self) -> int:
        return self.max - self.min

    def __str__(self) -> str:
        return f'Statistic(min={self.min},max={self.max},min_time={self.min_time}' \
               f',max_time={self.max_time},dif={self.dif},pvalue={self.pvalue})'

    def __repr__(self) -> str:
        return self.__str__()


def calc_freq_on_interval(neuron_activity: np.ndarray, start_time: np.ndarray, end_time: np.ndarray) -> np.ndarray:
    return (np.searchsorted(neuron_activity, end_time) - np.searchsorted(neuron_activity, start_time)) / \
           (end_time - start_time)


def calc_freq(neuron_activity: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    starts = np.arange(start_time, end_time - 1, STEP_DURATION)
    ends = starts + 1
    return calc_freq_on_interval(neuron_activity, starts, ends)[:PLOT_SIZE]


def process_apnoe(neuron_activity: np.ndarray, apnoe_time: float) -> np.ndarray:
    return calc_freq(neuron_activity, apnoe_time - APNOE_LEFT_DURATION, apnoe_time + APNOE_RIGHT_DURATION)


def process_plot(plot: np.ndarray) -> Statistic:
    result = Statistic()
    cur_time = -APNOE_LEFT_DURATION
    for point in plot:
        result.update(point, cur_time)
        cur_time += STEP_DURATION
    return result


def check_bootstrap_sample(sample: np.ndarray, point: float) -> bool:
    return np.abs(sample - point).min() > BOOTSTRAP_PROHIBITED_TIME


def generate_bootstrap_sample(max_time: float, size: int) -> np.ndarray:
    done = 0
    sample = np.zeros(size)
    while done < size:
        new_sample = sps.uniform(0, max_time - APNOE_DURATION).rvs(size=size - done)
        for point in new_sample:
            if 0 == done or check_bootstrap_sample(sample[:done], point):
                sample[done] = point
                done += 1
    return sample


def process_neuron(neuron_activity: np.ndarray, apnoe_times: np.ndarray) -> Statistic:
    apnoes = []
    for apnoe_time in apnoe_times:
        apnoes.append(process_apnoe(neuron_activity, apnoe_time))
    apnoes = np.array(apnoes).mean(axis=0)  # заменяем несколько графиков на их среднее
    apnoe_stats = process_plot(apnoes)
    bootstrap_stats = np.zeros(BOOTSTRAP_SIZE)
    for i in range(BOOTSTRAP_SIZE):
        bootstrap_plots = [
            calc_freq(neuron_activity, start, start + APNOE_DURATION)
            for start in generate_bootstrap_sample(neuron_activity[-1] - APNOE_DURATION, len(apnoe_times))
        ]
        bootstrap_plots = np.array(bootstrap_plots).mean(axis=0)
        bootstrap_stats[i] = process_plot(bootstrap_plots).dif
    bootstrap_stats.sort()
    apnoe_stats.pvalue = (BOOTSTRAP_SIZE - np.searchsorted(bootstrap_stats, apnoe_stats.dif)) / BOOTSTRAP_SIZE
    return apnoe_stats


def load_file(name: str) -> np.ndarray:
    return np.loadtxt(name, skiprows=5, unpack=True)


def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'


def main(root_path: str):
    apnoes: tp.Optional[np.ndarray] = None
    neuron_activities: tp.Dict[str, np.ndarray] = {}
    for file_name in os.listdir(root_path):
        if len(file_name) > 4 and '.txt' == file_name[-4:]:
            if 'Apnoe.txt' == file_name or 'Neuron' == file_name[:len('Neuron')]:
                print(f'Loading {file_name}...', end='\t')
                if 'Apnoe.txt' == file_name:
                    apnoes = load_file(os.path.join(root_path, 'Apnoe.txt'))
                else:
                    neuron_activities.update({
                        file_name[:-4]: load_file(os.path.join(root_path, file_name))
                    })
                print('OK')
    print('apnoe_times:', list(apnoes))
    for neuron, neuron_activity in neuron_activities.items():
        sys.stderr.flush()
        sys.stdout.flush()
        print(neuron)
        print(process_neuron(neuron_activity, apnoes))


if '__main__' == __name__:
    main(os.path.dirname(os.path.abspath(__file__)))

