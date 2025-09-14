class InnovationTracker:
    """Globally tracks new structural innovations (connections and nodes)."""
    def __init__(self):
        self.counter = 0
        self.history = {}

    def get_innovation(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        if key not in self.history:
            self.counter += 1
            self.history[key] = self.counter
        return self.history[key]