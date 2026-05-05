import pandas as pd
import random
import numpy as np
import json
import warnings
import math
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
def generate_random_strategy(feature_info, max_rules=3):
    num_features = len(feature_info)
    rules_matrix = []
    tps = []
    sls = []

    for r in range(max_rules):
        rule = [info["dont_care"] for info in feature_info]
        # انتخاب تصادفی ۱ تا ۵ شرط فعال برای جلوگیری از تولید قوانین غیرممکن
        num_active_conditions = random.randint(3, 5) 
        active_indices = random.sample(range(num_features), num_active_conditions)
        
        for idx in active_indices:
            info = feature_info[idx]
            rule[idx] = random.randint(0, info["num_classes"] - 1)
            
        rules_matrix.append(rule)
        tps.append(round(random.uniform(0.5, 5.0), 2))
        sls.append(round(random.uniform(0.5, 5.0), 2))

    return rules_matrix, tps, sls


# =========================================================
# ۳. نگاشت اسمی (تولید خروجی خوانا)
# =========================================================
def decode_to_human_readable(rules_matrix, tps, sls, feature_info, fintess_rules_values, strategy_score, rule_counter_arr, trade_counter, direction="long"):
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
                "score": fintess_rules_values[r_idx],
                "trade_count": rule_counter_arr[r_idx],
                "conditions": rule_conditions
            })

    best_strategy = {
        "direction": direction,
        "score": strategy_score,
        "trade_count": trade_counter,
        "rules_set": rules_set
    }
    
    return best_strategy


def fitness(df, rules, tps, sls, feature_info, direction):
    '''
    in this function we check if the trade that open used that
    strategy is in trend direction or not using max_befor_min column.
    also check if strategy open enough trade (50).
    '''
    counter = 0
    finish_all_rules_money = 0
    score_arr = []
    rule_counter_arr = []
    #print(rules)
    print('evaluate fitness...')

    for idx, rule in enumerate(rules):
        finish_each_rules_money = 0
        rule_counter = 0
        for index, row in df.iterrows(): # loop each row in csv file
            if check_rule(row, rule, feature_info): # check if row has this rules                
                money = compute_money(row, is_tp_trriger(row, tps[idx], direction), is_sl_trriger(row, sls[idx], direction), tps[idx], sls[idx], direction)
                finish_each_rules_money = finish_each_rules_money + money
                finish_all_rules_money = finish_all_rules_money + money
                rule_counter = rule_counter + 1
                counter = counter + 1

        score_arr.append(finish_each_rules_money)
        rule_counter_arr.append(rule_counter)

    return score_arr, finish_all_rules_money, rule_counter_arr, counter



def check_rule(row, rules, feature_info):
    '''
    check if given row constrain given rule or not
    if row constrain given rule return True else False
    '''
    for idx, rule_value in enumerate(rules):
        dont_care_value = feature_info[idx]["dont_care"]

        #print('rule:', rule_value, ', dont care value:', dont_care_value)
        # اگر rule_value == -1 باشد یعنی dont_care (از مقدار منفی استفاده کردیم)
        if rule_value == dont_care_value:
            #print('continue')
            continue
            

        #col_min = df[feature_info[idx]['col']].min()
        #col_max = df[feature_info[idx]['col']].max()

        #print('col_min:', col_min, 'col_max:', col_max, 'num_classes:', feature_info[idx]['num_classes'], 'mode:', feature_info[idx]['mode'])
            
        actual_value_raw = row[feature_info[idx]['col']]
        actual_value = normalize_feature_value(actual_value_raw, feature_info[idx])
        #print('actual_value_raw:', actual_value_raw, 'actual value:', actual_value, 'rule value:', rule_value, 'num_classes:', feature_info[idx]['num_classes'], 'mode:', feature_info[idx]['mode'])

        if actual_value != rule_value:
            #print('actual_value_raw:', actual_value_raw, 'actual value:', actual_value, 'rule value:', rule_value, 'num_classes:', feature_info[idx]['num_classes'], 'mode:', feature_info[idx]['mode'])
            return False
    
    #print('return true')
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
        if -1<value <= -0.6:
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
        if -1<value <= -0.8:
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


def is_tp_trriger(item, tp, direction):
    '''
    check if tp trigger in this row or not
    return True if trigger else False
    '''
    if direction == 'long':
        # حد سود در معامله خرید: قیمت باید بره بالا
        max_price = item.get('label_max_288', 0)
        entry = item.get('entry_price', item.get('label_open_next', 0))
        if entry > 0:
            return ((max_price - entry) / entry * 100) >= tp
    else:  # short
        # حد سود در معامله فروش: قیمت باید بره پایین
        min_price = item.get('label_min_288', 0)
        entry = item.get('entry_price', item.get('label_open_next', 0))
        if entry > 0:
            return ((entry - min_price) / entry * 100) >= tp
    return False


def is_sl_trriger(item, sl, direction):
    '''
    check if sl trigger in this row or not
    return True if trigger else False
    '''
    if direction == 'long':
        # حد ضرر در معامله خرید: قیمت باید بره پایین
        min_price = item.get('label_min_288', 0)
        entry = item.get('entry_price', item.get('label_open_next', 0))
        if entry > 0:
            return ((entry - min_price) / entry * 100) >= sl
    else:  # short
        # حد ضرر در معامله فروش: قیمت باید بره بالا
        max_price = item.get('label_max_288', 0)
        entry = item.get('entry_price', item.get('label_open_next', 0))
        if entry > 0:
            return ((max_price - entry) / entry * 100) >= sl
    return False


def compute_money(row, is_tp_trigger, is_sl_trigger, tp, sl, direction):
    '''
    compute the profit and lose in this row
    return the profit or loss as number
    '''
    max_before_min = row.get('label_max_before_min')
    
    if is_tp_trigger and not is_sl_trigger:
        return tp
    
    if is_sl_trigger and not is_tp_trigger:
        return -sl
    
    if not is_sl_trigger and not is_tp_trigger:
        open_price = row.get('label_open_next')
        close_price = row.get('label_close_288', 0)
        if math.isnan(open_price) or math.isnan(close_price):
            return 0
        
        money = ((close_price - open_price) / close_price) * 100
        if math.isnan(money):
            print('money:', money)
            print(row.get('label_close_288'), row.get('label_open_next'))
        
        if direction == 'long':
            return money
        
        return -money 
    
    if direction == 'long' and max_before_min == 0:
        return tp
    elif direction == 'long' and max_before_min == 1:
        return -sl
    
    if direction == 'short' and max_before_min == 1:
        return tp
    elif direction == 'short' and max_before_min == 0:
        return -sl




def genetic(population, crossover_rate, mutation_rate):
    '''
    implement genetic algorithm
    pop is array in size of population size
    '''
    new_rules = []
    new_tps = []
    new_sls = []

    population_scores = population['rules_score']
    population_counters = population['rule_counter_arr']
    parents_proability = proability(population_scores, population_counters)
    first_selected_parents_indexs, second_selected_parents_indexs = parent_selection(parents_proability)

    for i in range(len(first_selected_parents_indexs)):
        individuals = [population['rules'][first_selected_parents_indexs[i]], population['rules'][second_selected_parents_indexs[i]]]
        individuals_proabilities = [parents_proability[first_selected_parents_indexs[i]], parents_proability[second_selected_parents_indexs[i]]]
        individuals_tps = [population['tps'][first_selected_parents_indexs[i]], population['tps'][second_selected_parents_indexs[i]]]
        individuals_sls = [population['sls'][first_selected_parents_indexs[i]], population['sls'][second_selected_parents_indexs[i]]]

        crossover_rule, crossover_tp, crossover_sl = crossover(individuals, individuals_proabilities, individuals_tps, individuals_sls, crossover_rate)
        new_rule, new_tp, new_sl = mutation(crossover_rule, crossover_tp, crossover_sl, mutation_rate)

        new_rules.append(new_rule)
        new_tps.append(new_tp)
        new_sls.append(new_sl)

    return new_rules, new_tps, new_sls


def mutation(rule, tp, sl, mutation_rate):
    '''
    given a array like [5, 5, 5, 5, 10, 5, 5, 10, 5, 2, 5, 5, 5, 0, 5, 10, 5, 5, 5, 5, 10, 10, 5, 5, 5, 10, 5, 2, 5, 5, 5, 10, 5, 5, 10, 10, 0, 10, 10, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    and tp like 4.95 and sl like 3.53
    with rate of mutation_rate, mutate the array and tp and sl
    return mutated array, tp, sl
    '''
    mutated_arr = []
    mutated_tp = 0
    mutated_sl = 0

    return mutated_arr, mutated_tp, mutated_sl


def crossover(individuals, individuals_proabilities, individuals_tps, individuals_sls, crossover_rate):
    '''
    in this function given individuals and
    individuals_proabilities and individuals_tps and individuals_sls.

    base on the individuals_proabilities[0] and individuals_proabilities[1]
    this function must crossover individuals[0] and individuals[1] and select tps and sls.
    
    the given elements are like:
    individual[0] = [5, 5, 5, 5, 10, 5, 5, 10, 5, 2, 5, 5, 5, 0, 5, 10, 5, 5, 5, 5, 10, 10, 5, 5, 5, 10, 5, 2, 5, 5, 5, 10, 5, 5, 10, 10, 0, 10, 10, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    individual[1] = [5, 5, 5, 5, 10, 5, 5, 2, 5, 3, 5, 5, 5, 5, 5, 10, 5, 0, 5, 5, 10, 10, 5, 5, 5, 10, 5, 0, 5, 5, 5, 10, 5, 5, 10, 10, 10, 10, 10, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    individuals_proabilities[0] = 0.65682
    individuals_proabilities[1] = 0.178363

    return the crossover array of individuals[0] and individuals[1] and tp and sl
    '''

    tp = 0
    sl = 0
    new_individual =[]
    
    return new_individual, tp, sl


def proability(scores, counts):
    '''
    evaluate the proability of each element base on the 
    scores array and counts array

    the scores and counts are like below but in diffrent length but
    the length of score and counts are the same and each one represent
    a parent. for example score[0] and counts[0] represent 
    score and count of trade of parent 1:
    score = [75.94368743237938, 0, 2099.665335045179, -142.76935036714227, 158.95332184296228]
    counts = [122, 0, 5124, 4971, 653]

    return a array that each element is between 0 and 1. each
    element of aaray represent the proability of each parent.
    '''

    proabilites = []

    return proabilites


def parent_selection(probability):
    '''
    select parent base on the proability each have.
    the probability is a array that every elements of it represent
    one parent proability. it look like: [0.2, 0.6543, 0.09875, 0.86542, 0.37652, 0.15426, 0.237653]
    return the index of the parents as arrays like: [10, 4, 65, 35], [5, 23, 64, 44]
    '''
    parent1 = []
    parent2 = []

    return parent1, parent2


# =========================================================
# ۴. اجرای برنامه و تولید فایل نهایی
# =========================================================
if __name__ == "__main__":

    direction = 'long'
    pop_size = 5
    generation=100
    crossover_rate = 0.9
    mutation_rate = 0.01

    print("Loading data to extract feature info...")
    # ⚠️ آدرس فایل دیتاست خود را در این قسمت وارد کنید
    df = pd.read_csv("../data/test-v2.csv")
    print('is nan:', df['label_close_288'].isna().any())
    

    
    label_cols = ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
    meta_cols = ["datetime", "symbol"]
    feature_cols = [c for c in df.columns if c not in label_cols + meta_cols]
    
    # استخراج متادیتا ویژگی‌ها
    feature_info = get_features_info(df, feature_cols)
    

    #=============================================
    # Making initial population
    #=============================================
    # تولید قوانین عددی تصادفی (شما این بخش را با هوش مصنوعی خود جایگزین می‌کنید)
    best_rules, best_tps, best_sls = generate_random_strategy(feature_info, pop_size)

    # evaluate the generated strategy using train data
    rules_score, strategy_score, rule_counter_arr, trade_counter = fitness(df, best_rules, best_tps, best_sls, feature_info, direction)

    population = {
        'rules': best_rules,
        'tps': best_tps,
        'sls': best_sls,
        'rules_score': rules_score,
        'strategy_score': strategy_score,
        'rule_counter_arr': rule_counter_arr,
        'trade_counter': trade_counter
    }


    #=============================================
    # Start Genetic algorithm
    #=============================================

    for _ in generation:
        new_rules, new_tps, new_sls = genetic(population, crossover_rate, mutation_rate)

        rules_score, strategy_score, rule_counter_arr, trade_counter = fitness(df, new_rules, new_tps, new_sls, feature_info, direction)
        
        population = {
            'rules': best_rules,
            'tps': best_tps,
            'sls': best_sls,
            'rules_score': rules_score,
            'strategy_score': strategy_score,
            'rule_counter_arr': rule_counter_arr,
            'trade_counter': trade_counter
        }
        









    print(population)
    
    # ترجمه مقادیر عددی به رشته‌های خوانا 
    final_human_readable_strategy = decode_to_human_readable(
        best_rules, best_tps, best_sls, feature_info, rules_score, strategy_score, rule_counter_arr, trade_counter, direction
    )

    #print(final_human_readable_strategy)
    
    
    # ذخیره در فایل JSON
    output_filename = "short.json"
    with open(output_filename, "a", encoding="utf-8") as f:
        json.dump(final_human_readable_strategy, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Strategy generated and saved to '{output_filename}'")
    print("=" * 50)
    print("Preview of your submission format:\n")
    print(json.dumps(final_human_readable_strategy, indent=4))
    