#!/usr/bin/env python
import os
import sys
import pika
import json
import mysql.connector  # pip install mysql-connector-python
import requests


class TelegramSenderBuilder:
    def __init__(self):
        print("create Telegram Sender")
        # self.chat_ids = {"5847068700", "866464798"}
        self.chat_ids = {"5847068700"}
        self.TOKEN = "6196626152:AAEtnpMWTefy0QAbKOxIO682FpMshAjdF3Y"
        # self.chat_ids = []
        # self.update_chat_ids()
        self.message = "No data has been recorded"

    def update_chat_ids(self):
        url = f"https://api.telegram.org/bot{self.TOKEN}/getUpdates"
        updates = requests.get(url).json()
        num_updates = len(updates["result"])
        # last_update = num_updates - 1
        # text = updates["result"][last_update]["message"]["text"]
        # chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        for update_index in range(num_updates):
            chat_id = updates["result"][update_index]["message"]["chat"]["id"]
            self.chat_ids.append(chat_id)
        print("chat ids registered: ")
        print(self.chat_ids)

    def update_message(self, message):
        self.message = message
        self.send_message()

    def send_message(self):
        # chat id (Colin): 5847068700
        # chat id (Ana): 866464798
        for chat_id in self.chat_ids:
            url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage?chat_id={chat_id}&text={self.message}"
            print(requests.get(url).json())     # this sends the message


class SQLDatabaseBuilder:
    def __init__(self,
                 number_of_vacancies_in_lot,
                 number_of_cars_in_lot,
                 direction_of_vehicles_entering,
                 direction_of_vehicles_exiting):
        print("create SQL database builder")

        self.database_name = "db_deepstreamSolution"
        self.table_name = "table_deepstreamSolution"

        self.create_database()
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Password",
            database=self.database_name
        )

        self.number_of_vacancies_in_lot = number_of_vacancies_in_lot
        self.number_of_cars_in_lot = number_of_cars_in_lot
        self.direction_of_vehicles_entering = direction_of_vehicles_entering
        self.direction_of_vehicles_exiting = direction_of_vehicles_exiting

    def create_database(self):
        mydb_to_create = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Password"
        )
        mycursor = mydb_to_create.cursor(buffered=True)
        mycursor.execute("SHOW DATABASES;")

        carpark_database_created = False
        for x in mycursor:
            if x[0] == self.database_name:
                carpark_database_created = True
                break

        if not carpark_database_created:
            mycursor.execute("CREATE DATABASE " + self.database_name)
        mycursor.close()

    def create_table(self):
        mycursor = self.mydb.cursor(buffered=True)
        mycursor.execute("SHOW TABLES")

        table_created = False
        for x in mycursor:
            if x[0] == self.table_name:
                table_created = True
                break

        if not table_created:
            print("create table: " + self.table_name)
            mycursor.execute("CREATE TABLE " + self.table_name +
                             " (id int, va_filter_name VARCHAR(255), message_str VARCHAR(255))")

    def delete_table(self):
        mycursor = self.mydb.cursor(buffered=True)
        mycursor.execute("SHOW TABLES")

        table_created = False
        for x in mycursor:
            if x[0] == self.table_name:
                table_created = True
                break

        if table_created:
            print("delete table: " + self.table_name)
            sql = "DROP TABLE " + self.table_name
            mycursor.execute(sql)

    def get_number_of_entries(self):
        mycursor = self.mydb.cursor(buffered=True)
        mycursor.execute("SHOW TABLES")

        mycursor.execute("SELECT * FROM " + self.table_name)

        '''myresult = mycursor.fetchall()

        for x in myresult:
            print(x)

        print(mycursor.rowcount, "records inserted.")'''
        return mycursor.rowcount

    def convert_msg_string_to_dict(self, msg_string: str):
        return json.loads(msg_string)

    def write_to_table(self, va_output):
        mycursor = self.mydb.cursor()

        for filter_name in va_output:
            id = self.get_number_of_entries() + 1
            print(filter_name)
            sql = "INSERT INTO " + self.table_name + " (id, va_filter_name, message_str) VALUES (%s, %s, %s)"
            val = (id, filter_name, va_output[filter_name])
            mycursor.execute(sql, val)

            if filter_name == 'VehicleMonitorFilter':
                message_dict = self.convert_msg_string_to_dict(va_output[filter_name])
                self.update_vacancy(message_dict['direction'])

        self.mydb.commit()

    # the next 2 functions are specific to VehicleMonitorFilter. They should be moved later
    def update_vacancy(self, direction_of_vehicle):
        if direction_of_vehicle == self.direction_of_vehicles_entering:     # someone is entering the lot
            self.number_of_vacancies_in_lot -= 1
            self.number_of_cars_in_lot += 1
        if direction_of_vehicle == self.direction_of_vehicles_exiting:      # someone is exiting the lot
            self.number_of_vacancies_in_lot += 1
            self.number_of_cars_in_lot -= 1

    def get_vacancy(self):
        return self.number_of_vacancies_in_lot


class MessageProcessorBuilder:
    def __init__(self, sql_database, telegramSender):
        self.channel = None
        self.sql_database = sql_database
        self.telegramSender = telegramSender
        self.create_message_processor()
        self.start_message_processor()

    def create_message_processor(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = connection.channel()

        self.channel.queue_declare(queue='deepstreamSolution')

        def callback(ch, method, properties, body):
            print(" [x] Received %r" % body.decode())
            va_output = json.loads(body.decode())
            print(va_output)
            self.sql_database.write_to_table(va_output)

            vacancy = self.sql_database.get_vacancy()
            message = "No message. An error has occured"
            if vacancy <= 0:
                message = "Sorry, there are no lots available in the parking lot"
            elif vacancy == 1:
                message = "There is " + str(vacancy) + " lots in the parking lot"
            elif vacancy > 1:
                message = "There are " + str(vacancy) + " lots in the parking lot"
            self.telegramSender.update_message(message)

        self.channel.basic_consume(queue='deepstreamSolution', on_message_callback=callback, auto_ack=True)

    def start_message_processor(self):
        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()


def main():
    parking_lot_capacity = 20
    number_of_cars_in_lot = 19
    direction_of_vehicles_entering = 'left moving'
    direction_of_vehicles_exiting = 'right moving'

    sql_database = SQLDatabaseBuilder(parking_lot_capacity-number_of_cars_in_lot,
                                      number_of_cars_in_lot,
                                      direction_of_vehicles_entering,
                                      direction_of_vehicles_exiting)
    sql_database.create_database()
    sql_database.delete_table()
    sql_database.create_table()

    telegramSender = TelegramSenderBuilder()
    MessageProcessorBuilder(sql_database, telegramSender)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
