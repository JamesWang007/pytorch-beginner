# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:16:38 2019


        

@author: bejin
"""

"""
argparse
    https://docs.python.org/3.6/library/argparse.html
"""
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args(['--sum', '7', '-1', '42'])
#args = parser.parse_args()
print(args.accumulate(args.integers))



# eg - 1

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-f', '--foo')
parser.add_argument('bar')
parser.add_argument('bar2')
args = parser.parse_args(['BAR'])
print(args)

# eg = 2
# action
parser = argparse.ArgumentParser()
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
#parser.parse_args([])
args = parser.parse_args('--mgpus'.split())
print(args)


# eg - 3
parser = argparse.ArgumentParser()
parser.add_argument('--foo', nargs=2)
parser.add_argument('bar', nargs=1)
parser.parse_args('c --foo a b'.split())


# eg - 4
parser = argparse.ArgumentParser()
parser.add_argument('--foo', nargs='?', const='c', default='d')
parser.add_argument('bar', nargs='?', default='d')
parser.parse_args(['XX', '--foo', 'YY'])

parser.parse_args(['XX', '--foo'])

parser.parse_args([])


# eg - 5
parser = argparse.ArgumentParser()
parser.add_argument('foo', type=int)
parser.add_argument('bar', type=open)
parser.parse_args('2 temp.txt'.split())

parser = argparse.ArgumentParser()
parser.add_argument('bar', type=argparse.FileType('w'))
parser.parse_args(['out.txt'])


# eg - 6
def perfect_square(string):
    value = int(string)
    sqrt = math.sqrt(value)
    if sqrt != int(sqrt):
        msg = "%r is not a perfect square" % string
        raise argparse.ArgumentTypeError(msg)
    return value

import math
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('foo', type=perfect_square)
parser.parse_args(['9'])

parser.parse_args(['7']) 



# eg - 7
parser = argparse.ArgumentParser(prog='game.py')
parser.add_argument('move', choices=['rock', 'paper', 'scissors'])
parser.parse_args(['rock'])

parser.parse_args(['fire']) 


# eg - 8
parser = argparse.ArgumentParser(prog='doors.py')
parser.add_argument('door', type=int, choices=range(1, 4))
print(parser.parse_args(['3']))

parser.parse_args(['4'])



# eg - 9 : metavar
parser = argparse.ArgumentParser()
parser.add_argument('--foo')
parser.add_argument('bar')
parser.parse_args('X --foo Y'.split())

parser.print_help()


parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-x', nargs=2)
parser.add_argument('--foo', nargs=2, metavar=('bar', 'baz'))
parser.print_help()
parser.parse_args('--foo XXX YYY'.split())


# eg - 10 : dest
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--foo-bar', '--foo')
parser.add_argument('-x', '-y')
parser.parse_args('-f 1 -x 2'.split())
parser.parse_args('--foo 1 -y 2'.split())


parser = argparse.ArgumentParser()
parser.add_argument('--foo', dest='bar')
parser.parse_args('--foo XXX'.split())




























