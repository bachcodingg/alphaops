"""End-to-end CLI tests: the same commands cron runs, with their exit codes."""

import json

import pytest

from alphaops.cli import EXIT_DEGRADED, EXIT_FAILED, EXIT_OK, main


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOPS_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("ALPHAOPS_METRICS_DIR", str(tmp_path / "metrics"))
    for var in (
        "ALPHAOPS_PUSHGATEWAY_URL",
        "ALPHAOPS_SLACK_WEBHOOK_URL",
        "ALPHAOPS_TELEGRAM_BOT_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    def run(*args):
        return main(["--log-format", "json", "--log-level", "ERROR", *args])

    return run


INGEST = [
    "ingest",
    "--source",
    "synthetic",
    "--n-symbols",
    "15",
    "--start",
    "2022-01-01",
    "--end",
    "2024-12-31",
    "--no-alert",
]


def test_full_workflow(cli, capsys, tmp_path):
    assert cli(*INGEST) == EXIT_OK
    assert "SUCCESS" in capsys.readouterr().out

    assert cli("simulate", "rank(-ts_sum(returns, 5))", "--decay", "3", "--json") == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["sharpe"] > 0 and out["settings"]["decay"] == 3

    pnl_csv = tmp_path / "pnl.csv"
    assert cli("simulate", "rank(volume)", "--pnl-out", str(pnl_csv)) == EXIT_OK
    assert pnl_csv.read_text().startswith("date,pnl,cum_pnl,turnover")

    assert cli("batch", "examples/alphas.toml", "--no-alert") == EXIT_OK
    assert "reversal_5d" in capsys.readouterr().out
    assert cli("alphas") == EXIT_OK
    assert cli("runs") == EXIT_OK
    assert "evaluate" in capsys.readouterr().out
    assert cli("dq") == EXIT_OK
    assert cli("ops") == EXIT_OK
    assert "ts_decay_linear" in capsys.readouterr().out
    assert (tmp_path / "metrics" / "alphaops_ingest.prom").exists()


def test_degraded_exit_code_and_quarantine_review(cli, capsys):
    args = [*INGEST, "--anomalies"]
    assert cli(*args) == EXIT_DEGRADED
    capsys.readouterr()
    assert cli("quarantine", "list") == EXIT_OK
    listing = capsys.readouterr().out
    first_id = int(listing.splitlines()[1].split()[0])
    assert cli("quarantine", "drop", str(first_id)) == EXIT_OK
    assert cli("quarantine", "drop", str(first_id)) == EXIT_FAILED  # already resolved
    assert cli("quarantine", "release") == EXIT_FAILED  # missing id


def test_health_exit_codes(cli, capsys):
    assert cli("health") == 2  # empty DB: CRITICAL
    capsys.readouterr()
    cli(*INGEST)
    capsys.readouterr()
    code = cli("health", "--json")
    report = json.loads(capsys.readouterr().out)
    # data ends 2024-12-31 so freshness is critical "today"; integrity must still be ok
    assert code == 2 and report["status"] == "critical"
    assert {c["name"]: c["status"] for c in report["checks"]}["db_integrity"] == "ok"


def test_bad_expression_and_missing_data(cli, capsys):
    assert cli("simulate", "rank(close") == EXIT_FAILED  # no data yet
    cli(*INGEST)
    capsys.readouterr()
    assert cli("simulate", "rank(close") == EXIT_FAILED
    assert "expected ',' or ')'" in capsys.readouterr().err


def test_csv_source_requires_dir(cli):
    assert cli("ingest", "--source", "csv", "--no-alert") == EXIT_FAILED


def test_usage_error_exit_code(cli):
    with pytest.raises(SystemExit) as exc:
        cli("ingest", "--start", "not-a-date")
    assert exc.value.code == 2
