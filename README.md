# Diplomacy AI Agents Archive

A small archive of the **Diplomacy** agents I developed alongside the baseline agents provided for the project.

This was the project for CITS3011: Intelligent Agents in 2025.


## 🎯 Project Goal

Develop an agent capable of outperforming the scenario where each opponent could be the following:

- `RandomAgent` 20%
- `AttitudeAgent` 40%
- `GreedyAgent` 40%

Testing was performed over **700 games**.


## 🧠 Performance Results

Results of my agent across 700 games (100 games as each power).

| Country  | Avg SCs | Std Dev | Wins | Survives | Defeats |
|---------:|--------:|--------:|-----:|---------:|--------:|
| AUSTRIA  | 10.20 | 7.33 | 43.0% | 49.0% | 8.0% |
| ENGLAND  | 8.56 | 3.62 | 1.0% | 96.0% | 3.0% |
| FRANCE   | 15.97 | 4.26 | 76.0% | 24.0% | 0.0% |
| GERMANY  | 15.78 | 4.70 | 76.0% | 23.0% | 1.0% |
| ITALY    | 7.65 | 6.59 | 24.0% | 69.0% | 7.0% |
| RUSSIA   | 14.45 | 5.80 | 69.0% | 29.0% | 2.0% |
| TURKEY   | 14.95 | 3.97 | 56.0% | 44.0% | 0.0% |
| **ALL**  | 12.51 | 6.29 | **49.29%** | **47.71%** | **3.0%** |

> **Overall**: ~50% win rate against the pool — strong dominance as France and Germany.


## 📝 About Environment

Diplomacy is a negotiation-focused strategy board game featuring seven major powers:
> Austria, England, France, Germany, Italy, Russia, and Turkey

Each turn is simultaneous, and success relies on forming (and betraying!) alliances. For more information, look into the rules of the game.


## 🍕 Resources

- Engine: https://github.com/diplomacy/diplomacy  
- Python Package Docs: https://diplomacy.readthedocs.io/en/stable/  
- Diplomacy Rules: https://en.wikibooks.org/wiki/Diplomacy/Rules  