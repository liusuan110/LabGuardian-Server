from app.domain.board_schema import BoardSchema


def test_default_board_schema_uses_63_row_split_rails() -> None:
    schema = BoardSchema.default_breadboard()

    assert schema.resolve_hole_to_node("LP1") == "TRACK_LP_SEG1"
    assert schema.resolve_hole_to_node("LP31") == "TRACK_LP_SEG1"
    assert schema.resolve_hole_to_node("LP32") == "TRACK_LP_SEG2"
    assert schema.resolve_hole_to_node("RN63") == "TRACK_RN_SEG2"

    expected_hole_count = (5 * 63) + (5 * 63) + (4 * 63) + 2
    assert len(schema.holes) == expected_hole_count


def test_default_board_schema_track_assignment_nodes_are_segmented() -> None:
    schema = BoardSchema.default_breadboard()

    assert schema.resolve_track_assignment_nodes("top_plus") == [
        "TRACK_LP_SEG1",
        "TRACK_LP_SEG2",
    ]

