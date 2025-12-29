from pathlib import Path


def generate_tree(pathname, n=0):
    if pathname.is_file():
        print('│ ' * n + '├── ' + pathname.name)
    elif pathname.is_dir():
        print('│ ' * n + '├── ' + pathname.name + '/')
        for cp in sorted(pathname.iterdir()):
            generate_tree(cp, n + 1)


if __name__ == '__main__':
    generate_tree(Path.cwd())
