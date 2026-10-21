from dj_score import __version__
from dj_score.cli import main


def test_version():
    assert __version__


def test_cli_help(capsys):
    main([])
    assert "dj-score" in capsys.readouterr().out
