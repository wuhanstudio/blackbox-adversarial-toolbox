import click

@click.group()
def api():
    """Manage Cloud APIs"""
    pass

@click.group()
def attack():
    """Manage Attacks"""
    pass

@click.group()
def example():
    """Manage Examples"""
    pass

@click.group()
def main_cli():
    """The CLI tool for Black-box Adversarial Toolbox (BAT)."""
    pass

@api.command('list')
def api_list():
    """List supported Cloud APIs"""
    click.echo('List supported Cloud APIs')

@attack.command('list')
def attack_list():
    """List supported Attacks"""
    click.echo('List supported Attacks')

@example.command('list')
def example_list():
    """List available examples"""
    click.echo('List examples')

def main():
    main_cli.add_command(api)
    main_cli.add_command(attack)
    main_cli.add_command(example)

    api.add_command(api_list)

    attack.add_command(attack_list)

    example.add_command(example_list)

    return main_cli()

if __name__ == "__main__":

    main()
