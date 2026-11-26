"""Schema from BlueTable's initial Blue observation, never the simulator truth table."""


def visible_host_schema(wrapper):
    from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper

    # Read exactly the wrapper layer that serializes the actor's 52-value vector.
    blue = wrapper
    while not isinstance(blue, BlueTableWrapper):
        if not hasattr(blue, "env"):
            raise ValueError("The installed wrapper has no visible Blue table")
        blue = blue.env
    names, groups, subnet_ids = [], {}, []
    for name, row in blue.info.items():
        if len(row) != 5 or row[2] != name:
            raise ValueError("Unexpected actor-visible Blue table schema")
        names.append(name)
        # Numeric IPs/subnets are randomized. Retain only visible group membership,
        # labeled by first occurrence in this fixed actor-visible host ordering.
        subnet_ids.append(groups.setdefault(row[0], len(groups)))
    if len(names) != 13 or len(groups) != 3:
        raise ValueError("Host-aware v1 expects the verified 13-host, three-subnet interface")
    return tuple(names), tuple(subnet_ids)
