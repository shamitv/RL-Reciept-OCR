from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text


def test_normalize_text() -> None:
    assert normalize_text("  Hello   World ") == "HELLO WORLD"


def test_normalize_amount() -> None:
    assert normalize_amount("31") == "31.00"
    assert normalize_amount("TOTAL 31.0") == "31.00"
    assert normalize_amount("Sales Tax (9.5%) $0.83") == "0.83"
    assert normalize_amount("Tax 8.75% 1.11") == "1.11"
    assert normalize_amount("TAX(9%) : 2.80") == "2.80"
    assert normalize_amount("TOTAL EURO *17 .00 TOTAL TTC 10% *17.00") == "17.00"
    assert normalize_amount("13-Jul-2018 8:53:47P") == ""
    assert normalize_amount("Bill Date 06.07.2017 Bill Time 22:06") == ""


def test_normalize_date() -> None:
    assert normalize_date("25/03/2019") == "2019-03-25"
    assert normalize_date("05/11/18/ 8:47 PM") == "2018-11-05"
    assert normalize_date("Printed By: Rebecca B Jun 26, 15 04:55 PM ID: 234673 #1") == "2015-06-26"
    assert normalize_date("Date!: Apr 01, 2019 Time: 05:12PM Open Time : Apr 01, 2019 04:39PM") == "2019-04-01"
    assert normalize_date("13-Jul-2018 8:53:47P") == "2018-07-13"
    assert normalize_date("0168 20:14 #04 MAR. 19 ' 17 REG0001") == "2017-03-19"
    assert normalize_date("Date: Jan 28, 2018 5:48:07 PM") == "2018-01-28"
    assert normalize_date("Bill Date 06.07.2017 Bill Time 22:06") == "2017-07-06"
    assert normalize_date("25AUG ' 17 8:03PM") == "2017-08-25"
    assert normalize_date("Order Started: 11:01 AM") == ""


def test_normalize_address() -> None:
    assert normalize_address("12 Road\nKL") == "12 ROAD KL"
