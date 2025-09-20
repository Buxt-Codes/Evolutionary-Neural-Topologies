class Config:

    # =====================
    # Database
    # =====================
    # Population
    num_islands = 5
    max_pop_per_island = 100
    initial_pop_per_island = 25

    # Feature Bins
    max_nodes_bin = 1000
    max_layers_bin = 1000
    num_nodes_bins = 10
    num_layers_bins = 10

    # Network
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

    # Evaluatation
    epochs = 3
    batch_size = 32
    train_data_path = ""
    val_data_path = ""

    # Initialisation
    initial_connection_prob = 0.5
    initial_max_hidden = 50
    initial_max_layers = 10