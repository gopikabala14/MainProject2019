import sqlite3
import time
import datetime
import random

conn = sqlite3.connect('toll_data.db')
c = conn.cursor()


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS toll(datestamp TEXT,time TEXT, car TEXT, bus TEXT, truck TEXT, amount REAL)")


def data_entry():
    c.execute("INSERT INTO toll VALUES(1452549219,'2016-01-11 13:53:39','Python',6)")

    conn.commit()
    # c.close()
    # conn.close()


def dynamic_data_entry():
    unix = int(time.time())
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
    car='car'
    bus='bus'
    truck='truck'
    amount=100

    c.execute("INSERT INTO toll(datestamp, car,bus,truck,amount) VALUES (?, ?, ?, ?,?)",
              (date, car,bus,truck,amount))

    conn.commit()



create_table()
#data_entry()
dynamic_data_entry()


c.close
conn.close()