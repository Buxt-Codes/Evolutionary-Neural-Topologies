class Config:

    # =====================
    # Evolutionary Loop
    # =====================
    max_iterations = 1000
    num_workers = 10
    worker_gpu_fraction = 1 / num_workers * 0.75
    worker_timeout = 300
    log_path = "evolution.log"
    stats_path = "stats.csv"

    exploitation_prob = 0.2

    # =====================
    # Database
    # =====================
    db_path = "database.pkl"
    db_save_interval = 5

    # Population
    num_islands = 5
    max_population = 250
    initial_pop_per_island = 10

    # Feature Bins
    max_nodes_bin = 1000
    max_layers_bin = 1000
    num_nodes_bins = 10
    num_layers_bins = 10

    # Migration
    migration_interval = 10
    migration_prob = 0.5
    num_to_migrate = 10 # Ensure that the num_to_migrate x num_islands <= max_population

    # =====================
    # Network
    # =====================
    # Network
    input_size = 9
    output_size = 1

    # Bias
    use_bias = True

    # Mutation
    max_mutations = 20
    prob_add_connection = 0.3
    prob_remove_connection = 0.3
    prob_add_node = 0.15
    prob_remove_node = 0.15

    # Crossover
    crossover_prob = 0.5

    # Initialisation
    initial_connection_prob = 0.5
    initial_max_hidden = 25
    initial_max_layers = 20

    # =====================
    # Evaluation
    # =====================
    epochs = 3
    batch_size = 1024
    train_data_path = "data/processed/train.csv"
    val_data_path = "data/processed/val.csv"

    model_path = "trained_models/"

