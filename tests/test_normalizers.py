from env.normalizers import normalize_address, normalize_amount, normalize_date, normalize_text


def test_normalize_text() -> None:
    assert normalize_text("  Hello   World ") == "HELLO WORLD"


def test_normalize_amount() -> None:
    assert normalize_amount("31") == "31.00"
    assert normalize_amount("TOTAL 31.0") == "31.00"


def test_normalize_date() -> None:
    assert normalize_date("25/03/2019") == "2019-03-25"
    assert normalize_date("05/11/18/ 8:47 PM") == "2018-11-05"


def test_normalize_address() -> None:
    assert normalize_address("12 Road\nKL") == "12 ROAD KL"
