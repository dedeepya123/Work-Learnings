

def build_gen_ai_config_dict(qc_config, rope_theta: float | None = None) -> dict:
    if rope_theta is None:
        rope_theta = qc_config.rope_parameters["sliding_attention"]["rope_theta"]

    config_dict = qc_config.to_dict()
    config_dict["rope_theta"] = rope_theta
    return config_dict
