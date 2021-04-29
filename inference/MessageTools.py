import json
from termcolor import colored
from datetime import date
from datetime import datetime as dt
from tabulate import tabulate




def show_err(msg, override=False):
    msg = '[' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + ']\t' + '[ERROR]\t' + msg
    print(colored(msg, 'red')) # print to terminal
    if override: return

def show_succ(msg, override=False):
    msg = '[' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + ']\t' + msg
    print(colored(msg, 'green')) # print to terminal
    if override: return

def show_blue(msg, override=False):
    msg = '[' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + ']\t' + msg
    print(colored(msg, 'blue')) # print to terminal
    if override: return

def show_yellow(msg, override=False):
    msg = '[' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + ']\t' + msg
    print(colored(msg, 'yellow')) # print to terminal
    if override: return

def show_white(msg, override=False):
    msg = '[' + dt.now().strftime('%Y-%m-%d %H:%M:%S') + ']\t' + msg
    print(msg)
    if override: return
