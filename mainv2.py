import pandas as pd
import random
import numpy as np
import json
import warnings
import math
import copy
warnings.filterwarnings("ignore")


# =========================================================
# ۱. توابع تشخیص نوع ویژگی‌ها (Feature Modes)
# =========================================================
def detect_feature_mode(series):
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)
    if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
        return "binary"
    if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
        return "ternary"
    
    zero_ratio = (series == 0).mean()
    if series.min() < 0:
        return "sparse_signed" if zero_ratio > 0.3 else "signed"
    else:
        return "sparse_positive" if zero_ratio > 0.3 else "positive"


def get_features_info(df, feature_cols):
    feature_info = []
    for col in feature_cols:
        mode = detect_feature_mode(df[col])
        if mode == "binary":
            feature_info.append({"col": col, "mode": mode, "num_classes": 2, "dont_care": 2})
        elif mode == "ternary":
            feature_info.append({"col": col, "mode": mode, "num_classes": 3, "dont_care": 3})
        elif mode in ["positive", "sparse_positive"]:
            feature_info.append({"col": col, "mode": mode, "num_classes": 5, "dont_care": 5})
        elif mode == "sparse_signed":
            feature_info.append({"col": col, "mode": mode, "num_classes": 5, "dont_care": 5})
        else: # signed
            feature_info.append({"col": col, "mode": mode, "num_classes": 10, "dont_care": 10})
    return feature_info


# =========================================================
# ۲. تولید استراتژی تصادفی (به صورت عددی)
# =========================================================
def generate_random_strategy(feature_info, max_rules=6):
    num_features = len(feature_info)
    rules_matrix = []
    tps = []
    sls = []
    cps = []

    for r in range(random.randint(2, max_rules)):
        rule = [info["dont_care"] for info in feature_info]
        # انتخاب تصادفی ۱ تا ۵ شرط فعال برای جلوگیری از تولید قوانین غیرممکن
        num_active_conditions = random.randint(1, 5) 
        active_indices = random.sample(range(num_features), num_active_conditions)
        
        for idx in active_indices:
            info = feature_info[idx]
            rule[idx] = random.randint(0, info["num_classes"] - 1)
            
        rules_matrix.append(rule)
        tps.append(round(random.uniform(1, 5.0), 2))
        sls.append(round(random.uniform(1, 5.0), 2))
        cps.append(round(random.uniform(5, 20.0), 0))

    return {
        "conditions": rules_matrix,   # List of rules
        "tps": tps,
        "sls": sls,
        "cps": cps
    }

# =========================================================
# ۳. Initial Population
# =========================================================
def generate_population(size, feature_info):

    print(f"Generating initial population of {size} strategies...")

    population = []

    for i in range(size):
        strategy = generate_random_strategy(feature_info)
        strategy["id"] = i
        population.append(strategy)

    return population


# =========================================================
# ۴. RULE CHECKING (for one rule)
# =========================================================
def check_rule(row, rules, feature_info):
    '''
    check if given row constrain given rule or not
    if row constrain given rule return True else False
    '''
    for idx, rule_value in enumerate(rules):
        dont_care_value = feature_info[idx]["dont_care"]

        if rule_value == dont_care_value:
            continue
            
        actual_value_raw = row[feature_info[idx]['col']]
        actual_value = normalize_feature_value(actual_value_raw, feature_info[idx])

        if actual_value != rule_value:
            return False
    
    return True


def normalize_feature_value(value, info):
    """
    Convert a raw feature value into the same discrete class space
    used by rules, based on feature mode (from get_features_info).
    """
    mode = info["mode"]
    num_classes = info["num_classes"]

    # --- Binary: values {0,1}
    if mode == "binary":
        return int(value)

    # --- Ternary: values {-1,0,1}
    if mode == "ternary":
        mapping = {-1: 0, 0: 1, 1: 2}  # class indices
        return mapping.get(int(value), 1)  # fallback to 0

    # --- Signed: equally divide full value range into bins
    if mode == "sparse_signed":
        if -1<=value<=-0.6:
            return 0
        elif -0.6<value<=-0.2:
            return 1
        elif -0.2<value<=0.2:
            return 2
        elif 0.2<value<=0.6:
            return 3
        elif 0.6<value<=1:
            return 4
    
    if mode == "signed":
        if -1<=value<=-0.8:
            return 0
        elif -0.8<value<=-0.6:
            return 1
        elif -0.6<value<=-0.4:
            return 2
        elif -0.4<value<=-0.2:
            return 3
        elif -0.2<value<=0:
            return 4
        elif 0<value <= 0.2:
            return 5
        elif 0.2<value<=0.4:
            return 6
        elif 0.4<value<=0.6:
            return 7
        elif 0.6<value<=0.8:
            return 8
        elif 0.8<value<=1:
            return 9

    # --- Positive or sparse_positive: map positive range 0..max into bins
    if mode in ["positive", "sparse_positive"]:
        if value <= 0.2:
            return 0
        elif 0.2<value<=0.4:
            return 1
        elif 0.4<value<=0.6:
            return 2
        elif 0.6<value<=0.8:
            return 3
        elif 0.8<value<=1:
            return 4


# =========================================================
# ۵. EVALUATE ONE STRATEGY (Full Match)
# =========================================================
def evaluate_strategy(df, strategy):
    """Evaluate one complete strategy (multiple rules)"""
    initial_equity = 1000
    equity = initial_equity
    profits = []
    wins = 0
    trade_count = 0
    peak = initial_equity
    max_drawdown = 0.0

    # Loop through each candle
    for _, row in df.iterrows():
        # Try each rule in the strategy (OR logic)
        for i, rule in enumerate(strategy["conditions"]):
            if check_rule(row, rule, feature_info):
                # Trade triggered by this rule
                tp = strategy["tps"][i]
                sl = strategy["sls"][i]
                cp = strategy["cps"][i]

                tp_hit = is_tp_hit(row, tp, DIRECTION)
                sl_hit = is_sl_hit(row, sl, DIRECTION)

                result = compute_trade(row, tp_hit, sl_hit, tp, sl, cp, equity)

                profits.append(result)
                equity += result
                trade_count += 1

                # === MAX DRAWDOWN IN PERCENTAGE ===
                if equity > peak:
                    peak = equity
                else:
                    current_dd = (peak - equity) / peak * 100
                    if current_dd > max_drawdown:
                        max_drawdown = current_dd

                if result > 0:
                    wins += 1

                break  # Only one trade per candle (first matching rule)

        if equity <= 10:  # Stop trading if equity is depleted
            break # Stop trading if equity is depleted

    # Penalize very few trades
    if trade_count < MIN_TRADES or equity <= 10:
        return {
            "fitness": -99999,
            "profit": -100,  # net profit
            "trades": trade_count,
            "winrate": 0.0,
            "drawdown": max_drawdown
        }
    # Metrics
    winrate = wins / trade_count if trade_count > 0 else 0
    avg_profit = np.mean(profits) if profits else 0
    std_profit = np.std(profits) if len(profits) > 1 else 0
    sharpe = avg_profit / std_profit if std_profit != 0 else 0

    fitness = (
        ((equity - initial_equity) / initial_equity) * 400 *
        winrate *
        np.log1p(trade_count) *
        max(0, 1 + sharpe)
    ) - (max_drawdown * 1.5)

    return {
        "fitness": fitness,
        "profit": ((equity - initial_equity) / initial_equity) * 100,  # net profit
        "trades": trade_count,
        "winrate": winrate,
        "drawdown": max_drawdown
    }


# ============================================================
# TP CHECK
# ============================================================

def is_tp_hit(row, tp, direction):

    entry = row["label_open_next"]

    if direction == "long":

        max_price = row["label_max_288"]

        move = (
            (max_price - entry)
            / entry
        ) * 100

        return move >= tp

    else:

        min_price = row["label_min_288"]

        move = (
            (entry - min_price)
            / entry
        ) * 100

        return move >= tp


# ============================================================
# SL CHECK
# ============================================================

def is_sl_hit(row, sl, direction):

    entry = row["label_open_next"]

    if direction == "long":

        min_price = row["label_min_288"]

        move = (
            (entry - min_price)
            / entry
        ) * 100

        return move >= sl

    else:

        max_price = row["label_max_288"]

        move = (
            (max_price - entry)
            / entry
        ) * 100

        return move >= sl
    


# ============================================================
# TRADE RESULT
# ============================================================

def compute_trade(row, tp_hit, sl_hit, tp, sl, cp, equity):

    max_before_min = row["label_max_before_min"]

    # TP only
    if tp_hit and not sl_hit:
        return (tp/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity

    # SL only
    if sl_hit and not tp_hit:
        return -sl/100 * cp / 100 * equity - FEE_PCT/100 * (cp / 100)*equity

    # Neither hit
    if not tp_hit and not sl_hit:

        open_price = row["label_open_next"]
        close_price = row["label_close_288"]

        if math.isnan(open_price) or math.isnan(close_price):
            return 0

        move = (
            (close_price - open_price)
            / open_price
        ) * 100

        if DIRECTION == "long":
            return (move/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity
        else:
            return (-move/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity

    # Both hit
    if DIRECTION == "long":

        if max_before_min == 1:
            return (tp/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity
        else:
            return (-sl/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity

    else:

        if max_before_min == 0:
            return (tp/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity
        else:
            return (-sl/100 * cp / 100)*equity - FEE_PCT/100 * (cp / 100)*equity

# =========================================================
# Population Evaluation
# =========================================================
def evaluate_population(df, population):
    results = []
    for i, strategy in enumerate(population):
        result = evaluate_strategy(df, strategy)
        result["id"] = strategy["id"]
        results.append(result)

        print(f"Strat {strategy['id']} | Fitness: {result['fitness']:8.1f} | "
              f"Profit: {result['profit']:8.1f}% | Trades: {result['trades']:3d} | "
              f"WR: {result['winrate']:6.1%} | DD: {result['drawdown']:6.1f}%")
    return results




# =========================================================
# GENETIC ALGORITHM FUNCTIONS
# =========================================================

def tournament_selection(population, results, k=5):
    """Select best individual from random k candidates"""
    selected_idx = random.sample(range(len(population)), k)
    best = max(selected_idx, key=lambda i: results[i]["fitness"])
    return population[best], results[best]


def crossover(parent1, parent2, feature_info):
    """Crossover between two strategies"""
    child = {"conditions": [], "tps": [], "sls": [], "cps": []}
    
    # Crossover rules
    num_rules = random.randint(2, 6)
    
    for _ in range(num_rules):
        if random.random() < 0.5 and parent1["conditions"]:
            rule = copy.deepcopy(random.choice(parent1["conditions"]))
            tp = random.choice(parent1["tps"])
            sl = random.choice(parent1["sls"])
            cp = random.choice(parent1["cps"])
        else:
            rule = copy.deepcopy(random.choice(parent2["conditions"]))
            tp = random.choice(parent2["tps"])
            sl = random.choice(parent2["sls"])
            cp = random.choice(parent2["cps"])
        
        child["conditions"].append(rule)
        child["tps"].append(tp)
        child["sls"].append(sl)
        child["cps"].append(cp)
    
    return child


def mutate(strategy, feature_info, mutation_rate=0.15):
    mutated = copy.deepcopy(strategy)
    
    # Mutate existing rules
    for i in range(len(mutated["conditions"])):
        rule = mutated["conditions"][i]
        
        if random.random() < mutation_rate and feature_info:
            idx = random.randint(0, len(rule)-1)
            if random.random() < 0.4:
                rule[idx] = feature_info[idx]["dont_care"]   # deactivate
            else:
                rule[idx] = random.randint(0, feature_info[idx]["num_classes"] - 1)
        
        # Mutate TP / SL / CP
        if random.random() < mutation_rate:
            mutated["tps"][i] = round(random.uniform(0.8, 6.0), 2)
        if random.random() < mutation_rate:
            mutated["sls"][i] = round(random.uniform(0.8, 6.0), 2)
        if random.random() < mutation_rate:
            mutated["cps"][i] = int(random.uniform(5, 60))
    
    # Add new rule safely
    if random.random() < 0.12 and len(mutated["conditions"]) < 8 and feature_info:
        new_strat = generate_random_strategy(feature_info)
        mutated["conditions"].append(new_strat["conditions"][0])
        mutated["tps"].append(new_strat["tps"][0])
        mutated["sls"].append(new_strat["sls"][0])
        mutated["cps"].append(new_strat["cps"][0])
    
    # Remove rule
    elif random.random() < 0.12 and len(mutated["conditions"]) > 2:
        idx = random.randint(0, len(mutated["conditions"])-1)
        del mutated["conditions"][idx]
        del mutated["tps"][idx]
        del mutated["sls"][idx]
        del mutated["cps"][idx]
    
    return mutated

# =========================================================
# MAIN GENETIC ALGORITHM
# =========================================================
def run_genetic_algorithm(df, feature_info, feature_cols, 
                         population_size,
                         generations,
                         mutation_rate,
                         direction,
                         min_trades):
    
    # Initialize population
    population = generate_population(population_size, feature_info)
    
    best_overall = None
    best_fitness = -float('inf')
    
    print(f"Starting GA with {population_size} individuals for {generations} generations...\n")
    
    for gen in range(generations):

        print(f"\n"+"===" * 10 + f" Generation {gen+1} " + "===" * 10)
        # Evaluate current population
        results = evaluate_population(df, population)
        
        # Find best in this generation
        gen_best_idx = np.argmax([r["fitness"] for r in results])
        gen_best_fitness = results[gen_best_idx]["fitness"]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_overall = copy.deepcopy(population[gen_best_idx])
            print(f"→ New Best at Generation {gen+1} | Fitness: {best_fitness:.1f}")
        
        # Create next generation
        next_population = []
        
        # Elitism: Keep top 2
        sorted_idx = sorted(range(len(results)), key=lambda i: results[i]["fitness"], reverse=True)
        next_population.append(copy.deepcopy(population[sorted_idx[0]]))
        next_population.append(copy.deepcopy(population[sorted_idx[1]]))
        
        # Fill the rest with crossover + mutation
        while len(next_population) < population_size:
            parent1, _ = tournament_selection(population, results, k=5)
            parent2, _ = tournament_selection(population, results, k=5)
            
            child = crossover(parent1, parent2, feature_info)
            child = mutate(child, feature_info, mutation_rate)
            child["id"] = len(next_population)
            next_population.append(child)
        
        population = next_population
        
        if (gen + 1) % 10 == 0:
            print(f"Generation {gen+1}/{generations} completed | Best Fitness: {best_fitness:.1f}")
    
    print("\n" + "="*70)
    print("GENETIC ALGORITHM FINISHED!")
    print("="*70)
    
    return best_overall


# =========================================================
# ۳. نگاشت اسمی (تولید خروجی خوانا)
# =========================================================
def decode_to_human_readable(rules_matrix, tps, sls, cps, feature_info):
    """
    این تابع آرایه‌های عددی را به ساختار دیکشنری با نام‌های خوانا تبدیل می‌کند.
    """
    rules_set = []
    
    for r_idx, rule in enumerate(rules_matrix):
        rule_conditions = []
        for idx, gene in enumerate(rule):
            info = feature_info[idx]
            
            # اگر ژن بی اهمیت (Don't Care) بود، از آن می‌گذریم
            if gene == info["dont_care"]:
                continue

            col = info["col"]
            mode = info["mode"]

            # تبدیل عدد به کلمه معادل بر اساس نوع ویژگی
            if mode == "binary":
                val = "Active (1)" if gene == 1 else "Inactive (0)"
            elif mode == "ternary":
                val = ["Negative (-1)", "Neutral (0)", "Positive (1)"][gene]
            elif mode in ["positive", "sparse_positive"]:
                val = ["Very Low", "Low", "Medium", "High", "Very High"][gene]
            elif mode == "sparse_signed":
                val = ["Strong Negative", "Weak Negative", "Exactly Zero", "Weak Positive", "Strong Positive"][gene]
            else: # signed
                val = ["Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish", "Neutral Negative", "Neutral Positive", "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish"][gene]

            rule_conditions.append(f"[{col}] IS {val}")

        # فقط قوانینی که شرط فعال دارند را اضافه می‌کنیم
        if rule_conditions:
            rules_set.append({
                "tp": tps[r_idx],
                "sl": sls[r_idx],
                "capital_pct": cps[r_idx],
                "conditions": rule_conditions
            })

    best_strategy = {
        "direction": DIRECTION,
        "rules_set": rules_set
    }
    
    return best_strategy


# =========================================================
# ۵. اجرای برنامه و تولید فایل نهایی
# =========================================================
if __name__ == "__main__":

    DIRECTION = 'long'
    POPULATION_SIZE = 10
    GENERATION=10
    MIN_TRADES = 10
    FEE_PCT = 0.2
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.1
    ELITE_PERCENT = 0.2

    print("Loading data to extract feature info...")
    # ⚠️ آدرس فایل دیتاست خود را در این قسمت وارد کنید
    df = pd.read_csv("./data/train2.csv")
    
    label_cols = ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
    meta_cols = ["datetime", "symbol"]
    feature_cols = [c for c in df.columns if c not in label_cols + meta_cols]
    
    # استخراج متادیتا ویژگی‌ها
    feature_info = get_features_info(df, feature_cols)    

    #=============================================
    # Making initial population
    #=============================================
    # تولید قوانین عددی تصادفی (شما این بخش را با هوش مصنوعی خود جایگزین می‌کنید)
    best_strategy = run_genetic_algorithm(
    df=df,
    feature_info=feature_info,
    feature_cols=feature_cols,
    population_size=POPULATION_SIZE,
    generations=GENERATION,
    mutation_rate=MUTATION_RATE,
    direction=DIRECTION,
    min_trades=MIN_TRADES
    )

    # Extract best components
    best_strategy_rules = best_strategy["conditions"]
    best_strategy_tps   = best_strategy["tps"]
    best_strategy_sls   = best_strategy["sls"]
    best_strategy_cps   = best_strategy["cps"]
    

    final_human_readable_strategy = decode_to_human_readable(
        best_strategy_rules, best_strategy_tps, best_strategy_sls, best_strategy_cps, feature_info
    )

    # ذخیره در فایل JSON
    output_filename = f"{DIRECTION}_not_tested.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_human_readable_strategy, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Strategy generated and saved to '{output_filename}'")
    print("=" * 50)
    print("Preview of your submission format:\n")
    #print(json.dumps(final_human_readable_strategy, indent=4))
    
