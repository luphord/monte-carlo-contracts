#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Composable financial contracts with Monte Carlo valuation
'''

__author__ = '''luphord'''
__email__ = '''luphord@protonmail.com'''
__version__ = '''0.1.0'''


from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description=__doc__)
parser.add_argument('--version',
                    help='Print version number',
                    default=False,
                    action='store_true')

subparsers = parser.add_subparsers(title='subcommands', dest='subcommand',
                                   help='Available subcommands')

mycmd_parser = subparsers.add_parser('mycmd', help='An example subcommand')
mycmd_parser.add_argument('-n', '--number',
                          help='some number',
                          default=17, type=int)


def _mycmd(args: Namespace) -> None:
    print('Running mycmd subcommand with n={}...'.format(args.number))
    print('mycmd completed')


mycmd_parser.set_defaults(func=_mycmd)


def main() -> None:
    args = parser.parse_args()
    if args.version:
        print('''Monte Carlo Contracts ''' + __version__)
    elif hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
