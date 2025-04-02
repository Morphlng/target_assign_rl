import json
import random
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from target_assign_rl.target_assign_aec import TaskAllocationAEC


class EnhancedGeneticAlgorithm:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.population_size = self.config.get("population_size", 100)
        # Increased generations to handle more complex scenarios
        self.generations = self.config.get("generations", 100)
        self.mutation_rate = self.config.get("mutation_rate", 0.3)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        self.tournament_size = self.config.get("tournament_size", 5)
        self.elitism_count = self.config.get("elitism_count", 3)
        self.food_supply = self.config.get("food_supply", 20)
        self.food_full = self.config.get("food_full", 0.7)
        self.prob_options = self.config.get("prob_options", [0.0, 0.2, 0.4, 0.6, 0.8])
        self.windows = self.config.get("windows", 4)
        self.individual = self.create_individual()

    def create_individual(self) -> List[float]:
        return [
            random.gauss(1.0, 0.5) if random.random() < 0.8 else random.uniform(0, 2)
            for _ in range(self.windows)
        ]

    def allocate_food(self, w: List[float], p: List[float]) -> List[int]:
        scores = []
        for i in range(self.windows):
            if p[i] == 0:
                scores.append(-np.inf)
            else:
                scores.append(w[i] * p[i])

        priority_order = sorted(range(self.windows), key=lambda x: -scores[x])
        allocation = [0] * self.windows
        remaining = self.food_supply

        # At least one food allocation for each window with non-zero probability
        for idx in priority_order:
            if p[idx] == 0:
                continue
            if remaining <= 0:
                break
            allocation[idx] += 1
            remaining -= 1

        # Optimizing the remaining food allocation
        while remaining > 0:
            marginal_gains = []
            for i in priority_order:
                if p[i] == 0:
                    continue
                current = allocation[i]
                marginal = (self.food_full**current) * self.food_full * p[i]
                marginal_gains.append((marginal, i))
            if not marginal_gains:
                break
            best = max(marginal_gains, key=lambda x: x[0])
            allocation[best[1]] += 1
            remaining -= 1

        return allocation

    def evaluate_individual(
        self, individual: List[float], p: List[float], x: List[int]
    ) -> float:
        allocation = self.allocate_food(individual, p)
        expected_coverage = 0.0
        for i in range(self.windows):
            if x[i] == 1:
                fi = allocation[i]
                success_prob = 1 - ((1 - self.food_full) ** fi)
                expected_coverage += success_prob
        used_food = sum(allocation)
        food_efficiency = used_food / self.food_supply
        fitness = expected_coverage * 0.5 + food_efficiency * 0.5
        return fitness

    def tournament_selection(
        self, population: List[List[float]], fitness: List[float]
    ) -> List[List[float]]:
        selected = []
        for _ in range(len(population)):
            participants = random.sample(
                list(zip(population, fitness)),
                min(self.tournament_size * 2, len(population)),
            )
            participants.sort(key=lambda x: -x[1])
            if random.random() < self.food_full:
                selected.append(participants[0][0])
            else:
                selected.append(random.choice(participants[1:3])[0])
        return selected

    def crossover(
        self, parent1: List[float], parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        if random.random() < 0.5:
            mix_mask = [
                random.random() < self.crossover_rate for _ in range(self.windows)
            ]
            child1 = [
                p1 if mask else p2 for p1, p2, mask in zip(parent1, parent2, mix_mask)
            ]
            child2 = [
                p2 if mask else p1 for p1, p2, mask in zip(parent1, parent2, mix_mask)
            ]
        else:
            point = random.randint(5, 15)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual: List[float]) -> List[float]:
        for i in range(self.windows):
            if random.random() < self.mutation_rate:
                if random.random() < 0.8:
                    individual[i] = np.clip(
                        individual[i] + random.gauss(0, 0.2), 0.1, 2.0
                    )
                else:
                    individual[i] = random.gauss(1.0, 0.5)
        return individual

    def save_model(self, individual: List[float], filename: str):
        with open(filename, "w") as f:
            json.dump(individual, f)

    def load_model(self, filename: str):
        with open(filename, "r") as f:
            self.individual = json.load(f)

    def train(self, env: "TaskAllocationAEC", model_path: str = None):
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = -np.inf
        history = []

        for generation in range(self.generations):
            env.reset()
            p = env.threat_levels
            x = env.actual_threats
            fitness = [self.evaluate_individual(ind, p, x) for ind in population]

            current_best_idx = np.argmax(fitness)
            current_fitness = fitness[current_best_idx]
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_individual = population[current_best_idx].copy()
                history.append(best_fitness)
                print(f"Gen {generation:03d} | Fitness: {best_fitness:.2f} ★")
            else:
                history.append(np.mean(fitness))
                print(f"Gen {generation:03d} | Avg: {np.mean(fitness):.2f}")

            selected = self.tournament_selection(population, fitness)
            next_population = []
            elite = sorted(zip(fitness, population), key=lambda x: -x[0])[
                : self.elitism_count
            ]
            next_population.extend([e[1].copy() for e in elite])

            while len(next_population) < self.population_size:
                parents = random.sample(selected, 2)
                child1, child2 = self.crossover(parents[0], parents[1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)

            population = next_population[: self.population_size]

        if model_path is not None:
            self.save_model(best_individual, model_path)
        else:
            self.save_model(best_individual, "model_path.pkl")
            print("Model path not provided, best individual saved model_path.pkl.")
        print("Training completed with final best fitness:", best_fitness)

    def predict(self, p: List[float]) -> List[int]:
        return self.allocate_food(self.individual, p)

    def update(self, p: List[float], x: List[int]):
        new_population = self.initialize_population_from_best(self.individual)
        fitness = [self.evaluate_individual(ind, p, x) for ind in new_population]
        current_best_idx = np.argmax(fitness)
        current_fitness = fitness[current_best_idx]
        best_fitness = self.evaluate_individual(self.individual, p, x)
        if current_fitness > best_fitness:
            self.individual = new_population[current_best_idx].copy()

    def initialize_population_from_best(self, best_individual):
        """
        使用部分随机初始化来生成新种群：
        - 种群的一半（或一部分）通过对 best_individual 进行变异得到
        - 其余个体随机生成，保证多样性
        """
        new_population = []
        # 设定变异个体的数量（例如：种群的一半）
        num_mutated = self.population_size // 2
        # 剩余部分完全随机生成
        num_random = self.population_size - num_mutated

        # 保留最佳个体（作为精英）
        new_population.append(best_individual.copy())

        # 生成基于最佳个体变异的个体（注意：已加入一个最佳个体，所以生成 num_mutated - 1 个变异个体）
        for _ in range(num_mutated - 1):
            mutated = self.mutate(best_individual)
            new_population.append(mutated)

        # 生成随机个体
        for _ in range(num_random):
            random_individual = self.create_individual()
            new_population.append(random_individual)

        # 可选：打乱新种群的顺序
        random.shuffle(new_population)
        return new_population
