from whales.viscous_drag import ViscousDragModel
import numpy as np

def test_model_loading():
    """Test simple model with one member"""
    config = {
        'drag coefficient': 0.7,
        'members': [{
            'end1': [0, 0, 0],
            'end2': [0, 0, -10],
            'diameter': [[0, 3, 5, 10], [2.2, 2.2, 5, 5]],
            'strip width': 0.5,
        }],
    }
    m = ViscousDragModel(config)

    # Check model setup correctly
    assert np.allclose(m.element_lengths, 0.5 * np.ones(20))
    assert np.allclose(m.element
