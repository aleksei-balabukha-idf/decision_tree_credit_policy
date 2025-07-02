import pandas as pd
import numpy as np
from typing import Set, Optional

class DecisionSplitter:
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        blocked_vars: Optional[Set[str]] = None,
        depth: int = 0,
        max_depth: int = 3,
        min_samples_split: int = 20,
        node_id: str = "0",  # Новая система идентификации узлов
        parent_info: str = ""
    ):
        self.data = data
        self.target = target
        self.blocked_vars = set(blocked_vars) if blocked_vars else set()
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.node_id = node_id
        self.parent_info = parent_info
        self.split_var = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.mean = data[target].mean()
        self.n_samples = len(data)

        self._log_node_creation()

    def _log_node_creation(self):
        indent = "  " * self.depth
        print(f"\n{indent}🟢 Узел {self.node_id} (глубина {self.depth}):")
        print(f"{indent}   • Образцы: {self.n_samples}")
        print(f"{indent}   • Средний target: {self.mean * 100 :.2f}%")
        if self.parent_info:
            print(f"{indent}   • Родительское правило: {self.parent_info}")
        if self.blocked_vars:
            print(f"{indent}   • Заблокированные переменные: {', '.join(self.blocked_vars)}")
        #print(f"{indent}{'-' * 40}")

    def find_best_split(self):
        best_gain = -np.inf
        best_var = None
        best_threshold = None

        candidate_vars = [
            col for col in self.data.columns 
            if col != self.target and col not in self.blocked_vars
        ]

        for var in candidate_vars:
            #print(f'var: {var}')
            unique_values = sorted(self.data[var].unique())
            for threshold in unique_values:
                left_data = self.data[self.data[var] <= threshold]
                right_data = self.data[self.data[var] > threshold]

                if len(left_data) < self.min_samples_split or len(right_data) < self.min_samples_split:
                    continue

                gain = self.calculate_split_gain(left_data, right_data)
                
                if gain > best_gain:
                    best_gain = gain
                best_var = var
                best_threshold = threshold

        return best_var, best_threshold, best_gain

    def calculate_split_gain(self, left_data, right_data):
        left_mean = left_data[self.target].mean()
        right_mean = right_data[self.target].mean()
        return abs(left_mean - right_mean)

    def split(self):
        list_all_cols = self.data.columns 
        list_blocked_cols = self.blocked_vars     
        list_vars_available = [i for i in list_all_cols if i not in list_blocked_cols]
        print(f'vars available: {list_vars_available}')
        if self.depth >= self.max_depth:
            self._log_max_depth_reached()
            return
        split_var, split_threshold, gain = self.find_best_split()
        
        if split_var is None:
            self._log_no_split_found()
            return

        self._log_best_split_found(split_var, split_threshold, gain)
        self.split_var = split_var
        self.split_threshold = split_threshold

        left_data = self.data[self.data[split_var] <= split_threshold]
        right_data = self.data[self.data[split_var] > split_threshold]

        new_blocked = set(self.blocked_vars)
        new_blocked.add(split_var)

        # Генерация ID для дочерних узлов
        left_id = self.node_id + "0"
        right_id = self.node_id + "1"

        left_info = f"{split_var} ≤ {split_threshold} → {left_id}"
        right_info = f"{split_var} > {split_threshold} → {right_id}"

        self.left = DecisionSplitter(
            left_data,
            self.target,
            blocked_vars=new_blocked,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            node_id=left_id,
            parent_info=left_info
        )

        self.right = DecisionSplitter(
            right_data,
            self.target,
            blocked_vars=new_blocked,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            node_id=right_id,
            parent_info=right_info
        )

        self.left.split()
        self.right.split()

    def _log_max_depth_reached(self):
        indent = "  " * self.depth
        print(f"{indent}⛔ Узел {self.node_id}: Достигнута максимальная глубина {self.max_depth}")

    def _log_no_split_found(self):
        indent = "  " * self.depth
        print(f"{indent}🔍 Узел {self.node_id}: Не найдено подходящего разделения")
        print(f"{indent}   • Минимальный размер узла: {self.min_samples_split}")
        print(f"{indent}   • Доступные переменные: {[col for col in self.data.columns if col != self.target and col not in self.blocked_vars]}")

    def _log_best_split_found(self, var, threshold, gain):
        indent = "  " * self.depth
        print(f"{indent}{'-' * 40}")
        print(f"{indent}🎯 Узел {self.node_id}: Лучшее разделение:")
        print(f"{indent}   • Переменная: {var}")
        print(f"{indent}   • Порог: {threshold}")
        print(f"{indent}   • Разница средних: {gain:.3f}")
        print(f"{indent}   • Левый потомок: {self.node_id}0")
        print(f"{indent}   • Правый потомок: {self.node_id}1")
