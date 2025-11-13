"""Sequence validation logic."""
import re
from typing import Dict, Iterable, List, Mapping

from flow_rules import FlowAssessment, assess_write_path

def get_tier_value(prime: int, schema: Dict[int, Dict]) -> int:
    """Converts tier string to a numeric value."""
    if not schema or prime not in schema:
        return 0
    tier_str = schema.get(prime, {}).get("tier", "")
    if not tier_str:
        return 0
    
    if "S!" in tier_str:
        return 1
    
    match = re.search(r'S(\d+)', tier_str)
    if match:
        return int(match.group(1))
        
    return 0

def validate_prime_sequence(sequence: List[Dict], schema: Dict[int, Dict]) -> bool:
    """
    Validates that the prime sequence does not contain illegal transitions.
    An illegal transition is from a lower tier to a higher tier.
    e.g. S0 -> S1 is illegal.
    """
    if len(sequence) < 2:
        return True

    for i in range(len(sequence) - 1):
        p1 = sequence[i].get("prime")
        p2 = sequence[i+1].get("prime")

        if p1 is None or p2 is None:
            continue

        if p1 == p2:
            continue

        tier1 = get_tier_value(p1, schema)
        tier2 = get_tier_value(p2, schema)

        if tier1 < tier2:
            print(f"[VALIDATION FAILED] Illegal transition from prime {p1} (Tier {tier1}) to {p2} (Tier {tier2})")
            return False

    return True


def assess_even_c_odd_path(
    entries: Iterable[Mapping[str, object] | int],
    schema: Mapping[int, Mapping[str, object]],
) -> FlowAssessment:
    """Return flow assessment for write paths enforcing even→C→odd sequencing."""

    return assess_write_path(entries, schema)
