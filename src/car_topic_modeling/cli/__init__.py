import typer
from . import data_cmd
from . import analysis_cmd

app = typer.Typer()
app.add_typer(data_cmd.app, name="data")
app.add_typer(analysis_cmd.app, name="analysis")
