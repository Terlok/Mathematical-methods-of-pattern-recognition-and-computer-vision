import os
import time
import json
import datetime
import numpy as np
from scipy import stats
from pynput import keyboard

class BiometricSystem:
    def __init__(self, reference_file, training_log_file, identification_log_file):
        self.reference_file = os.path.join(os.getcwd(), reference_file)
        self.training_log_file = os.path.join(os.getcwd(), training_log_file)
        self.identification_log_file = os.path.join(os.getcwd(), identification_log_file)
        self._ensure_files()

    def _ensure_files(self):
        if not os.path.exists(self.reference_file):
            with open(self.reference_file, 'w') as f:
                json.dump({}, f)

        if not os.path.exists(self.training_log_file):
            with open(self.training_log_file, 'w', encoding='utf-8') as f:
                f.write('timestamp;attempt;count;mean;variance\n')

        if not os.path.exists(self.identification_log_file):
            with open(self.identification_log_file, 'w', encoding='utf-8') as f:
                f.write('timestamp;attempt;test_mean;test_variance;alpha;verdict\n')

    @staticmethod
    def remove_outliers(data, m=3):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        return data[abs(data - mean) < m * std].tolist()

    @staticmethod
    def collect_intervals(control_phrase):
        input_times = []

        def on_press(key):
            input_times.append(time.time())
            if len(input_times) >= len(control_phrase):
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            typed_phrase = input()
            listener.join()

        if typed_phrase != control_phrase:
            print(f"Phrase doesn't match: '{control_phrase}'.")
            return []

        intervals = []
        for i in range(1, len(input_times)):
            intervals.append(input_times[i] - input_times[i - 1])

        return intervals


    @staticmethod
    def save_biometric_parameters(file_path, mean_val, var_val):
        reference_data = {'mean': mean_val, 'variance': var_val}
        with open(file_path, 'w') as f:
            json.dump(reference_data, f, indent=4)

    @staticmethod
    def load_biometric_parameters(file_path):
        if not os.path.exists(file_path):
            return None, None
        with open(file_path, 'r') as f:
            ref_data = json.load(f)
            return ref_data.get('mean'), ref_data.get('variance')

    @staticmethod
    def compare_statistics_classical(ref_mean, ref_var, test_mean, test_var, alpha=0.05):
        n_ref, n_test = 30, 30
        numerator = ref_mean - test_mean
        denominator = np.sqrt(ref_var / n_ref + test_var / n_test)
        T_calc = numerator / denominator if denominator != 0 else 0
        var_part = (ref_var / n_ref) + (test_var / n_test)
        numerator_df = var_part ** 2
        denominator_df = ((ref_var ** 2) / (n_ref ** 2 * (n_ref - 1))) + ((test_var ** 2) / (n_test ** 2 * (n_test - 1)))
        df_t = numerator_df / denominator_df if denominator_df != 0 else 1
        T_crit = stats.t.ppf(1 - alpha / 2, df_t)
        reject_mean = abs(T_calc) > T_crit
        F_calc = max(ref_var / test_var, test_var / ref_var) if test_var != 0 and ref_var != 0 else 1e8
        df1, df2 = n_ref - 1, n_test - 1
        F_crit_upper = stats.f.ppf(1 - alpha / 2, df1, df2)
        F_crit_lower = stats.f.ppf(alpha / 2, df1, df2)
        reject_var = (F_calc > F_crit_upper) or (F_calc < F_crit_lower)

        return not reject_mean and not reject_var

    def append_log(self, filename, line):
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(line)

    def training_flow(self):
        num_attempts = int(input('Enter the number of training attempts: '))
        control_phrase = input('Enter the control phrase: ').strip()

        all_intervals = []
        for attempt in range(1, num_attempts + 1):
            print(f'\n---\nTraining attempt {attempt}\n---')
            intervals = self.collect_intervals(control_phrase)
            intervals = self.remove_outliers(intervals)

            if intervals:
                mean_val = np.mean(intervals)
                var_val = np.var(intervals, ddof=1)
            else:
                mean_val = var_val = 0.0

            self.append_log(self.training_log_file,
                            f'{datetime.datetime.now()};{attempt};{len(intervals)};{mean_val:.5f};{var_val:.5f}\n')
            all_intervals.extend(intervals)

        all_intervals = self.remove_outliers(all_intervals)

        self.save_biometric_parameters(self.reference_file, np.mean(all_intervals), np.var(all_intervals, ddof=1))
        print('\nReference saved.\n')

    def identification_flow(self):
        num_attempts = int(input('Enter the number of identification attempts: '))
        alpha = float(input('Enter the significance level: '))
        control_phrase = input('Enter the control phrase: ').strip()

        ref_mean, ref_var = self.load_biometric_parameters(self.reference_file)
        if ref_mean is None or ref_var is None:
            print('No reference! Perform training first.')
            return

        success_count = 0
        for attempt in range(1, num_attempts + 1):
            print(f'\n---\nTraining attempt {attempt}\n---')
            intervals = self.collect_intervals(control_phrase)
            intervals = self.remove_outliers(intervals)

            test_mean = np.mean(intervals) if intervals else 0.0
            test_var = np.var(intervals, ddof=1) if intervals else 0.0

            identified = self.compare_statistics_classical(ref_mean, ref_var, test_mean, test_var, alpha)
            verdict = 'identified' if identified else '!identified'
            self.append_log(self.identification_log_file,
                            f'{datetime.datetime.now()};{attempt};{test_mean:.5f};{test_var:.5f};{alpha};{verdict}\n')

            if identified:
                success_count += 1

        print(f'\nSuccessful attempts: {success_count} out of {num_attempts}. \n'
              f'{'User identified\n' if success_count >= num_attempts / 2 else 'User not identified\n'}.')

    def run(self):
        while True:
            print('---\nTraining(T)\nIdentification(I)\nExit(E)\n---')
            choice = input()
            match choice:
                case 'T':
                    self.training_flow()
                case 'I':
                    self.identification_flow()
                case 'E':
                    print('Exiting the program.')
                    break
                case _:
                    print('Unknown command, please try again.')


if __name__ == '__main__':
    system = BiometricSystem('biometric_reference.json', 'training.csv', 'identification.csv')
    system.run()