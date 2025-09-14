# =====================
# CONFIG
# =====================

class Config:
    # Population
    pop_size = 150
    input_size = 4
    output_size = 1

    # Speciation
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4
    compatibility_threshold = 3.0

    # Mutation
    prob_add_connection = 0.1
    prob_add_node = 0.05
    prob_mutate_weight = 0.8
    weight_mutate_power = 0.1
    weight_replace_prob = 0.1

    # Reproduction
    survival_threshold = 0.2
    crossover_prob = 0.75

    # Initialisation
    initial_connection_prob = 0.5