"""Some of the capablities of the base env class
are moved here for better dynamic class definition
"""
from .render import get_render_method
from .take_action_kube import get_action_method_kube
from .take_action_sim import get_action_method_sim
from .step import get_step_method
from .reward import get_reward_method
