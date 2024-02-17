-- Keep a log of any SQL queries you execute as you solve the mystery.


-- To see list of tables in database
.tables

-- To see columns in crime scene reports
.schema crime_scene_reports

-- To see into crime_scene_report on day of crime
SELECT description FROM crime_scene_reports WHERE month=7 AND day=28 AND street='Humphrey Street';

-- Description: Theft of the CS50 duck took place at 10:15am at the Humphrey Street Bakery. Interviews were conducted today
-- with three witnesses who were present at the time - each of their interview transcripts mentions the bakery.
-- Littering took place at 16:36. No known witnesses.

-- To see columns in bakery_security_logs
.schema bakery_security_logs

-- To get license plate from bakery_security_logs between 10:15 and 10:25
SELECT license_plate from bakery_security_logs WHERE hour=10 AND minute BETWEEN 15 AND 25;

--+---------------+
--| N7M42GP       |
--| Y340743       |
--| 5P2BI95       |
--| 94KL13X       |
--| 6P58WS2       |
--| 4328GD8       |
--| G412CB7       |
--| L93JTIZ       |
--| 322W7JE       |
--| 0NTHK55       |
--| P14PE2Q       |
--| 1M92998       |
--| 11J91FW       |
--| PF37ZVK       |
--| 1M92998       |
--| XE95071       |
--| IH61GO8       |
--| 8P9NEU9       |
--+---------------+

-- To see columns in people table
.schema people

sqlite> SELECT * FROM people WHERE passport_number=3642612721;
--+--------+--------+----------------+-----------------+---------------+
--|   id   |  name  |  phone_number  | passport_number | license_plate |
--+--------+--------+----------------+-----------------+---------------+
--| 745650 | Sophia | (027) 555-1068 | 3642612721      | 13FNH73       |
--+--------+--------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=6264773605;
--+--------+---------+----------------+-----------------+---------------+
--|   id   |  name   |  phone_number  | passport_number | license_plate |
--+--------+---------+----------------+-----------------+---------------+
--| 341739 | Rebecca | (891) 555-5672 | 6264773605      | NULL          |
--+--------+---------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=6128131458;
--+--------+-------+----------------+-----------------+---------------+
--|   id   | name  |  phone_number  | passport_number | license_plate |
--+--------+-------+----------------+-----------------+---------------+
--| 423393 | Carol | (168) 555-6126 | 6128131458      | 81MNC9R       |
--+--------+-------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=7597790505;
--+--------+--------+----------------+-----------------+---------------+
--|   id   |  name  |  phone_number  | passport_number | license_plate |
--+--------+--------+----------------+-----------------+---------------+
--| 750165 | Daniel | (971) 555-6468 | 7597790505      | FLFN3W0       |
--+--------+--------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=1682575122;
--+--------+------+----------------+-----------------+---------------+
--|   id   | name |  phone_number  | passport_number | license_plate |
--+--------+------+----------------+-----------------+---------------+
--| 505688 | Jean | (695) 555-0348 | 1682575122      | JN7K44M       |
--+--------+------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=7179245843;
--+--------+-------+--------------+-----------------+---------------+
--|   id   | name  | phone_number | passport_number | license_plate |
--+--------+-------+--------------+-----------------+---------------+
--| 872102 | Joyce | NULL         | 7179245843      | 594IBK6       |
--+--------+-------+--------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=1618186613;
--+--------+--------+----------------+-----------------+---------------+
--|   id   |  name  |  phone_number  | passport_number | license_plate |
--+--------+--------+----------------+-----------------+---------------+
--| 632023 | Amanda | (821) 555-5262 | 1618186613      | RS7I6A0       |
--+--------+--------+----------------+-----------------+---------------+
sqlite> SELECT * FROM people WHERE passport_number=3835860232;
--+--------+--------+----------------+-----------------+---------------+
--|   id   |  name  |  phone_number  | passport_number | license_plate |
--+--------+--------+----------------+-----------------+---------------+
--| 780088 | Nicole | (123) 555-5144 | 3835860232      | 91S1K32       |
--+--------+--------+----------------+-----------------+---------------+


-- To see columns in passengers table
.schema passengers

-- To see all passengers leaving on flight ID 6
SELECT * FROM passengers WHERE flight_id=36;
--+-----------+-----------------+------+
--| flight_id | passport_number | seat |
--+-----------+-----------------+------+
--| 36        | 7214083635      | 2A   |
--| 36        | 1695452385      | 3B   |
--| 36        | 5773159633      | 4A   |****
--| 36        | 1540955065      | 5C   |
--| 36        | 8294398571      | 6C   |
--| 36        | 1988161715      | 6D   |
--| 36        | 9878712108      | 7A   |
--| 36        | 8496433585      | 7B   |
--+-----------+-----------------+------+


-- To see columns in airports table
.schema airports

-- To get id from flights leaving Boston
SELECT id FROM airports WHERE city='Fiftyville';

-- Fiftyville airport id: 8

-- To see columns in fligts
.schema flights

-- To get all flights leaving Fiftyville on the day after the crime
SELECT * FROM flights WHERE origin_airport_id=8 AND year=2021 AND month=7 AND day=29;

--+----+-------------------+------------------------+------+-------+-----+------+--------+
--| id | origin_airport_id | destination_airport_id | year | month | day | hour | minute |
--+----+-------------------+------------------------+------+-------+-----+------+--------+
--| 18 | 8                 | 6                      | 2021 | 7     | 29  | 16   | 0      |
--| 23 | 8                 | 11                     | 2021 | 7     | 29  | 12   | 15     |
--| 36 | 8                 | 4                      | 2021 | 7     | 29  | 8    | 20     |****
--| 43 | 8                 | 1                      | 2021 | 7     | 29  | 9    | 30     |
--| 53 | 8                 | 9                      | 2021 | 7     | 29  | 15   | 20     |
--+----+-------------------+------------------------+------+-------+-----+------+--------+

-- to see columns in interviews

.schema interviews

-- to get transcripts of witness interviews on day of crime

SELECT * FROM interviews WHERE month=7 AND day=28 AND year=2021;

-- | 161 | Ruth    | 2021 | 7     | 28  | Sometime within ten minutes of the theft, I saw the thief get into a car in the bakery parking lot and drive away. If you have security footage from the bakery parking lot, you might want to look for cars that left the parking lot in that time frame.                                                          |
-- | 162 | Eugene  | 2021 | 7     | 28  | I don't know the thief's name, but it was someone I recognized. Earlier this morning, before I arrived at Emma's bakery, I was walking by the ATM on Leggett Street and saw the thief there withdrawing some money.                                                                                                 |
-- | 163 | Raymond | 2021 | 7     | 28  | As the thief was leaving the bakery, they called someone who talked to them for less than a minute. In the call, I heard the thief say that they were planning to take the earliest flight out of Fiftyville tomorrow. The thief then asked the person on the other end of the phone to purchase the flight ticket. |

-- to see all columns in atm_transactions
.schema atm_transactions

-- to get all withdrawals on Leggett Street on day of crime
SELECT * FROM atm_transactions WHERE year=2021 AND day=28 AND month=7 AND atm_location="Leggett Street" AND transaction_type="withdraw";
--+-----+----------------+------+-------+-----+----------------+------------------+--------+
--| id  | account_number | year | month | day |  atm_location  | transaction_type | amount |
--+-----+----------------+------+-------+-----+----------------+------------------+--------+
--| 246 | 28500762       | 2021 | 7     | 28  | Leggett Street | withdraw         | 48     |
--| 264 | 28296815       | 2021 | 7     | 28  | Leggett Street | withdraw         | 20     |
--| 266 | 76054385       | 2021 | 7     | 28  | Leggett Street | withdraw         | 60     |
--| 267 | 49610011       | 2021 | 7     | 28  | Leggett Street | withdraw         | 50     |
--| 269 | 16153065       | 2021 | 7     | 28  | Leggett Street | withdraw         | 80     |
--| 288 | 25506511       | 2021 | 7     | 28  | Leggett Street | withdraw         | 20     |
--| 313 | 81061156       | 2021 | 7     | 28  | Leggett Street | withdraw         | 30     |
--| 336 | 26013199       | 2021 | 7     | 28  | Leggett Street | withdraw         | 35     |
--+-----+----------------+------+-------+-----+----------------+------------------+--------+

-- to see columns in phone_calls

.schema phone_calls

--to get all phone calls on date of crime lasting less than a minute

SELECT * FROM phone_calls WHERE year=2021 AND month=7 AND day=28 AND duration<=60;

--+-----+----------------+----------------+------+-------+-----+----------+
--| id  |     caller     |    receiver    | year | month | day | duration |
--+-----+----------------+----------------+------+-------+-----+----------+
--| 221 | (130) 555-0289 | (996) 555-8899 | 2021 | 7     | 28  | 51       |
--| 224 | (499) 555-9472 | (892) 555-8872 | 2021 | 7     | 28  | 36       |
--| 233 | (367) 555-5533 | (375) 555-8161 | 2021 | 7     | 28  | 45       |
--| 234 | (609) 555-5876 | (389) 555-5198 | 2021 | 7     | 28  | 60       |
--| 251 | (499) 555-9472 | (717) 555-1342 | 2021 | 7     | 28  | 50       |
--| 254 | (286) 555-6063 | (676) 555-6554 | 2021 | 7     | 28  | 43       |
--| 255 | (770) 555-1861 | (725) 555-3243 | 2021 | 7     | 28  | 49       |
--| 261 | (031) 555-6622 | (910) 555-3251 | 2021 | 7     | 28  | 38       |
--| 279 | (826) 555-1652 | (066) 555-9701 | 2021 | 7     | 28  | 55       |
--| 281 | (338) 555-6650 | (704) 555-2131 | 2021 | 7     | 28  | 54       |
--+-----+----------------+----------------+------+-------+-----+----------+
--to get all people leaving bakery around time of the crime
SELECT bakery_security_logs.activity, bakery_security_logs.license_plate, people.name FROM people JOIN bakery_security_logs ON bakery_security_logs.license_plate = people.license_plate WHERE bakery_security_logs.year=2021 AND bakery_security_logs.day=28 AND bakery_security_logs.hour=10 AND bakery_security_logs.minute BETWEEN 15 and 25;

--+----------+---------------+---------+
--| activity | license_plate |  name   |
--+----------+---------------+---------+
--| exit     | 5P2BI95       | Vanessa |
--| exit     | 94KL13X       | Bruce   |**
--| exit     | 6P58WS2       | Barry   |
--| exit     | 4328GD8       | Luca    |
--| exit     | G412CB7       | Sofia   |
--| exit     | L93JTIZ       | Iman    |
--| exit     | 322W7JE       | Diana   |**
--| exit     | 0NTHK55       | Kelsey  |
--+----------+---------------+---------+

-- to get names of people withdrawing money from ATM

SELECT people.name, atm_transactions.transaction_type FROM people JOIN bank_accounts ON bank_accounts.person_id = people.id JOIN atm_transactions ON atm_transactions.account_number = bank_accounts.account_number WHERE atm_transactions.year=2021 AND atm_transactions.month=7 AND atm_transactions.day=28 AND atm_location = "Leggett Street" AND atm_transactions.transaction_type = "withdraw";

--+---------+------------------+
--|  name   | transaction_type |
--+---------+------------------+
--| Bruce   | withdraw         |
--| Diana   | withdraw         |
--| Brooke  | withdraw         |
--| Kenny   | withdraw         |
--| Iman    | withdraw         |
--| Luca    | withdraw         |
--| Taylor  | withdraw         |
--| Benista | withdraw         |
--+---------+------------------+

--to get names with phone numbers
sqlite> SELECT phone_calls.caller, people.name FROM people JOIN phone_calls ON people.phone_number = phone_calls.caller WHERE phone_calls.year=2021 AND phone_calls.month=7 AND phone_calls.day=28 AND phone_calls.duration <=60;
--+----------------+---------+
--|     caller     |  name   |
--+----------------+---------+
--| (130) 555-0289 | Sofia   |
--| (499) 555-9472 | Kelsey  |
--| (367) 555-5533 | Bruce   |***
--| (609) 555-5876 | Kathryn |
--| (499) 555-9472 | Kelsey  |
--| (286) 555-6063 | Taylor  |
--| (770) 555-1861 | Diana   |***
--| (031) 555-6622 | Carina  |
--| (826) 555-1652 | Kenny   |
--| (338) 555-6650 | Benista |
--+----------------+---------+

sqlite> SELECT phone_calls.receiver, people.name FROM people JOIN phone_calls ON people.phone_number = phone_calls.receiver WHERE phone_calls.year=2021 AND phone_calls.month=7 AND phone_calls.day=28 AND phone_calls.duration <=60;
--+----------------+------------+
--|    receiver    |    name    |
--+----------------+------------+
--| (996) 555-8899 | Jack       |
--| (892) 555-8872 | Larry      |
--| (375) 555-8161 | Robin      |***Accomplice
--| (389) 555-5198 | Luca       |
--| (717) 555-1342 | Melissa    |
--| (676) 555-6554 | James      |
--| (725) 555-3243 | Philip     |
--| (910) 555-3251 | Jacqueline |
--| (066) 555-9701 | Doris      |
--| (704) 555-2131 | Anna       |
--+----------------+------------+
-- to get Bruce's passport #
SELECT * from people WHERE phone_number="(367) 555-5533";
--+--------+-------+----------------+-----------------+---------------+
--|   id   | name  |  phone_number  | passport_number | license_plate |
--+--------+-------+----------------+-----------------+---------------+
--| 686048 | Bruce | (367) 555-5533 | 5773159633      | 94KL13X       |
--+--------+-------+----------------+-----------------+---------------+
-- to get Diana's passport #
SELECT * from people WHERE phone_number="(770) 555-1861";
--+--------+-------+----------------+-----------------+---------------+
--|   id   | name  |  phone_number  | passport_number | license_plate |
--+--------+-------+----------------+-----------------+---------------+
--| 514354 | Diana | (770) 555-1861 | 3592750733      | 322W7JE       |
--+--------+-------+----------------+-----------------+---------------+
