from pathlib import Path

from analysis_v384_synthetic_data_report import build_report_text


def test_report_documents_authenticity_and_limits():
    text = build_report_text(
        {
            "v381_rows": 144,
            "top_candidate": "submission_v383_synth_scored_top9__v173action_v300server.csv",
            "top_point_churn": 9,
            "top_point0_additions": 0,
        }
    )

    assert "synthetic_" in text
    assert "No TTMATCH" in text
    assert "No hidden test labels" in text
    assert "submission_v383_synth_scored_top9__v173action_v300server.csv" in text


def test_report_writer_stays_inside_output_dir():
    from analysis_v384_synthetic_data_report import write_report

    outdir = Path("v384_synthetic_data_report") / "test_report_writer"
    path = write_report({"v381_rows": 1}, outdir=outdir)

    assert path.parent == outdir
    assert path.name == "synthetic_data_report.md"
