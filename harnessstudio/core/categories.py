"""The 12 harness categories, shared with RoboMetaHarness (knowledge/HARNESS_CATEGORIES.md)."""

CATEGORIES: dict[str, str] = {
    "observation_grounding": "Observation and visual grounding",
    "language_alignment": "Instruction grounding and rewriting",
    "proposal_selection": "Candidate sampling and selection",
    "sampler_guidance": "Inference-time action steering",
    "uncertainty_compute": "Uncertainty and inference allocation",
    "execution_monitoring": "Execution monitoring and recovery",
    "memory_retrieval": "Experience retrieval and transfer",
    "experience_evolution": "Code, prompt and memory evolution",
    "agentic_orchestration": "Planning and tool routing",
    "skill_composition": "Skill and controller composition",
    "world_model_verification": "Predicted-future and geometric verification",
    "contact_feedback": "Contact, grasp and slip feedback",
}
