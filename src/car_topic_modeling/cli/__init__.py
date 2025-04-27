import typer
from . import data_cmd

app = typer.Typer()
app.add_typer(data_cmd.app, name="data")
