# =====================
# CONFIG
# =====================

class Config:
    # Population
    pop_size = 150
    input_size = 4
    output_size = 1

    # Bias
    use_bias = True

    # Speciation
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4
    compatibility_threshold = 3.0

    # Mutation
    max_mutations = 1
    prob_add_connection = 0.1
    prob_remove_connection = 0.1
    prob_add_node = 0.05
    prob_remove_node = 0.05

    # Crossover
    prob_crossover = 0.8
    
    # Reproduction
    survival_threshold = 0.2
    crossover_prob = 0.75

    # Initialisation
    initial_connection_prob = 0.5
    initial_max_hidden = 50
    initial_max_layers = 10