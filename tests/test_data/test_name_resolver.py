import pandas as pd

from src.football_predictor.data.name_resolver import (
    build_name_lookup,
    resolve_team_name,
)


def test_build_name_lookup_and_resolve_team_name() -> None:
    former_names = pd.DataFrame(
        {
            "current": ["Brazil", "France"],
            "former": ["Brasil", "French Republic"],
            "start_date": ["2000-01-01", "2001-01-01"],
            "end_date": ["2005-01-01", "2006-01-01"],
        }
    )

    lookup = build_name_lookup(former_names)

    assert lookup["Brasil"] == "Brazil"
    assert resolve_team_name("Brasil", lookup) == "Brazil"
    assert resolve_team_name("Unknown", lookup) == "Unknown"
