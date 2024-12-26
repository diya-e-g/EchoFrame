# label_utils.py

def load_labels(label_path):
    """Load labels from a label map file."""
    with open(label_path, "r") as f:
        return [line.strip().split(" ", 1)[1] for line in f.readlines()]


def get_label_from_class_id(class_id, labels):
    """
    Convert class ID to a human-readable label using the label map.
    Assumes the model outputs 1-based class IDs.
    """
    return labels[class_id - 1]
