from moira.presentation_names import (
    routing_strategy_display_name,
    specialist_display_name,
    world_model_display_name,
)


def test_judge_facing_names_explain_specialist_roles():
    assert specialist_display_name("baseten-waypoint-policy") == "Movement Specialist"
    assert specialist_display_name("baseten-vision-scene") == "Vision Specialist"
    assert specialist_display_name("baseten-stt") == "Speech Recognition Specialist"
    assert specialist_display_name("baseten-task-reward") == "Action Scoring Specialist"


def test_simulation_and_router_terms_are_plain_language():
    assert world_model_display_name("rigid-dynamics") == "Rigid-Body Physics"
    assert routing_strategy_display_name("contract_singleton") == "Only compatible specialist"


def test_unknown_component_still_gets_a_readable_name():
    assert specialist_display_name("local-depth-estimator") == "Depth Estimator Specialist"
