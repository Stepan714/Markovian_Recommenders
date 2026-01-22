import polars as pl
from itertools import product

class MarkovChainPolars:
    def __init__(self, name=None, data=None, struct=None, time_column=None, users_id_column=None):
        # struct, for example (действие, id_song) = (название поля, название поля айдишника)
        self.NAME = name # String
        self.data = data # polars DataFrame
        self.struct = struct # tuple
        self.time_column = time_column # string
        self.users_id_column = users_id_column # string
        self.count_nodes = 0
        self.map_name_nodes: dict[int, str] = {}

        self.markov_chain: dict[str, dict[str, int]] = {}
        # {
        #     'id_node_1' : {2, 4, 6},
        #     'id_node_2': {1, 5, 70},
        #     ...
        #
        # }

    def preprocessing_data(self):
        if self.data is None:
            raise ValueError("Подан пустой датасет")

        self.data = self.data.sort([self.users_id_column, self.time_column])

        if not self.struct:
            self.count_nodes = 0
            self.map_name_nodes = {}
            return

        unique_values = [self.data[col].unique().to_list() for col in self.struct]
        unique_counts = [len(values) for values in unique_values]
        self.count_nodes = 1
        for count in unique_counts:
            self.count_nodes *= count

        self.map_name_nodes = {}
        node_id = 0

        for combination in product(*unique_values):
            key = "_".join(str(val) for val in combination)
            self.map_name_nodes[node_id] = key
            node_id += 1

        return

    def build_markov_chain(self):
        if self.data is None or not self.struct:
            return

        START_NODE = "START"  # Добавляем специальную стартовую ноду
        self.map_name_nodes[-1] = START_NODE  # Используем -1 как ID для стартовой ноды
        self.count_nodes += 1

        self.markov_chain = {  # Инициализация цепи с учетом стартовой ноды
            f'id_node_{i}': {}
            for i in range(-1, self.count_nodes - 1)  # Включаем -1 (старт) и обычные ноды
        }

        self.node_name_to_id = {v: k for k, v in self.map_name_nodes.items()}  # Создаем обратный маппинг для быстрого поиска

        # Группируем по пользователям и обрабатываем каждую группу
        grouped = self.data.group_by(self.users_id_column)
        
        for group in grouped:
            user_data = group
            prev_node = f'id_node_{-1}' # Всегда начинаем со стартовой ноды

            for row in user_data.iter_rows(named=True):
                current_values = [str(row[col]) for col in self.struct]
                current_node_key = "_".join(current_values)
                current_node_id = self.node_name_to_id[current_node_key]

                if f'id_node_{current_node_id}' in self.markov_chain[prev_node]:  # Добавляем переход
                    self.markov_chain[prev_node][f'id_node_{current_node_id}'] += 1
                else:
                    self.markov_chain[prev_node][f'id_node_{current_node_id}'] = 1

                prev_node = f'id_node_{current_node_id}'
                
        transitions = []
        for source, targets in self.markov_chain.items():
            for target, count in targets.items():
                transitions.append({
                    'Из': source,
                    'В': target,
                    'Переходы': count,
                    'Вероятность': f"{count / sum(targets.values()):.8%}"
                })

        self.P = pl.DataFrame(transitions).sort('Переходы', descending=True)        

        return

    def build_markov_chain_optimized(self):
        """
        Оптимизированная версия построения марковской цепи с использованием Polars.
        Использует оконные функции для нахождения переходов между состояниями.
        """
        # Если данные отсутствуют или структура не задана, прекращаем построение
        if self.data is None or not self.struct:
            return

        # Гарантируем корректный порядок событий: сортируем по пользователю и времени
        self.data = self.data.sort([self.users_id_column, self.time_column])

        # Специальная стартовая нода
        START_NODE = "START"

        # Переинициализируем маппинг: -1 соответствует START
        self.map_name_nodes = { -1: START_NODE }

        # Создаем колонку текущего состояния путем конкатенации всех колонок из struct
        data_with_states = self.data.with_columns(
            pl.concat_str([pl.col(col) for col in self.struct], separator="_").alias("current_state")
        )

        # Определяем уникальные состояния и их числовые идентификаторы
        unique_states = data_with_states["current_state"].unique().to_list()
        state_to_id = {state: idx for idx, state in enumerate(unique_states)}

        # Заполняем обратный маппинг: id -> name
        for state, node_id in state_to_id.items():
            self.map_name_nodes[node_id] = state

        # Общее количество узлов (+1 для START)
        self.count_nodes = len(state_to_id) + 1

        # Создаем DataFrame-мэппинг для добавления current_state_id
        mapping_df = pl.DataFrame({
            "current_state": list(state_to_id.keys()),
            "current_state_id": list(state_to_id.values()),
        })

        # Присоединяем ID состояния к основным данным
        data_with_states = (
            data_with_states
            .join(mapping_df, on="current_state", how="left")
            .with_columns(
                pl.col("current_state_id").cast(pl.Int64)
            )
        )

        # 1. Переходы внутри цепочек пользователя
        transitions = (
            data_with_states
            .group_by(self.users_id_column)
            .agg([
                pl.col("current_state_id").shift(1).alias("prev_state_id"),
                pl.col("current_state_id").alias("current_state_id"),
            ])
            .explode(["prev_state_id", "current_state_id"])
            .filter(pl.col("prev_state_id").is_not_null())
            .group_by(["prev_state_id", "current_state_id"])
            .agg(pl.count().alias("count"))
            .with_columns([
                pl.col("prev_state_id").cast(pl.Int64),
                pl.col("current_state_id").cast(pl.Int64),
                pl.col("count").cast(pl.Int64)
            ])
        )

        # 2. Переходы от START: первая точка входа в последовательность каждого пользователя
        start_transitions = (
            data_with_states
            .group_by(self.users_id_column)
            .agg(pl.col("current_state_id").first())
            .group_by("current_state_id")
            .agg(pl.count().alias("count"))
            .with_columns(pl.lit(-1).alias("prev_state_id"))
            .select(["prev_state_id", "current_state_id", "count"])
            .with_columns([
                pl.col("prev_state_id").cast(pl.Int64),
                pl.col("current_state_id").cast(pl.Int64),
                pl.col("count").cast(pl.Int64)
            ])
        )

        # 3. Объединяем оба типа переходов
        all_transitions = pl.concat([transitions, start_transitions])

        # 4. Создаем финальную таблицу: суммарные переходы и вероятности
        self.P = (
            all_transitions
            .with_columns([
                pl.format("id_node_{}", pl.col("prev_state_id")).alias("from_node_id"),
                pl.format("id_node_{}", pl.col("current_state_id")).alias("to_node_id")
            ])
            .group_by(["from_node_id", "to_node_id"])
            .agg(pl.sum("count").alias("Переходы"))
            .with_columns([
                ((pl.col("Переходы") / pl.col("Переходы").sum().over("from_node_id")) * 100.0).round(8).alias("_prob")
            ])
            .with_columns(
                pl.concat_str([pl.col("_prob").cast(pl.Utf8), pl.lit("%")]).alias("Вероятность")
            )
            .drop("_prob")
            .sort("Переходы", descending=True)
        )

        # Обратный маппинг имени узла в id
        self.node_name_to_id = {v: k for k, v in self.map_name_nodes.items()}

        return

    def filter_by_time_range(self, start_time: int, end_time: int):
        """
        Удаляет строки из self.data, где время не входит в указанный интервал.
        Время указывается в секундах (целые числа).
        
        :param start_time: нижняя граница времени (включительно)
        :param end_time: верхняя граница времени (включительно)
        """
        if self.data is None or self.time_column not in self.data.columns:
            raise ValueError("Нет данных или отсутствует колонка времени.")
    
        before_count = self.data.height
        self.data = self.data.filter(
            (pl.col(self.time_column) >= start_time) &
            (pl.col(self.time_column) <= end_time)
        )
        after_count = self.data.height
    
        print(f"Отфильтровано по времени: оставлено {after_count} из {before_count} записей "
              f"(границы: {start_time} - {end_time})")

    def show_markov_chain(self, visualize=True, top_n=10):
        # Красивый табличный вывод
        print(f"\nМарковская цепь '{self.NAME}' (первые {top_n} переходов):")
        print("=" * 60)

        # Создаем DataFrame для красивого отображения
        if hasattr(self, 'P') and self.P is not None:
            print(self.P.head(top_n))
        else:
            print("Марковская цепь еще не построена")