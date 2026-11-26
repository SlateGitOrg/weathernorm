from datetime import date

import pytest
from conftest import mini_connection, rendered

D = date

# One home whose retrofit is in June 2024. Reads are hand-built to hit every rule.
READS = [
    (1, D(2024, 1, 5), 1000.0, False),
    (1, D(2024, 2, 5), 1100.0, False),  # 31 days, 100 therms: accepted, pre
    (1, D(2024, 3, 5), 1150.0, True),  # estimated, merged into the next actual read
    (1, D(2024, 4, 4), 1350.0, False),  # 02-05 to 04-04: 59 days, 250 therms: too_long
    (1, D(2024, 4, 20), 1360.0, False),  # 16 days: too_short
    (1, D(2024, 5, 20), 1300.0, False),  # -60 therms: negative_consumption
    (1, D(2024, 6, 10), 1400.0, False),  # overlaps June: blackout
    (1, D(2024, 7, 10), 1500.0, False),  # still overlaps June (days from 06-11): blackout
    (1, D(2024, 8, 9), 1600.0, False),  # starts after June: accepted, post
]


@pytest.fixture(scope="module")
def mini():
    con = mini_connection([], [(1, "participant", D(2024, 6, 15))], READS)
    con.execute(rendered("010_billing_periods.sql"))
    return con


def test_estimated_read_is_merged_into_the_next_actual_read(mini):
    row = mini.execute(
        "SELECT start_date, end_date, days, therms, merged_estimated_reads "
        "FROM periods WHERE end_date = DATE '2024-04-04'"
    ).fetchone()
    assert row == (D(2024, 2, 5), D(2024, 4, 4), 59, 250.0, 1)


def test_reject_reasons_in_priority_order(mini):
    reasons = dict(mini.execute("SELECT end_date, reject_reason FROM periods").fetchall())
    assert reasons == {
        D(2024, 2, 5): None,
        D(2024, 4, 4): "too_long",
        D(2024, 4, 20): "too_short",
        D(2024, 5, 20): "negative_consumption",
        D(2024, 6, 10): "blackout",
        D(2024, 7, 10): "blackout",
        D(2024, 8, 9): None,
    }


def test_phases_are_assigned_only_to_accepted_periods(mini):
    phases = dict(mini.execute("SELECT end_date, phase FROM periods").fetchall())
    assert phases[D(2024, 2, 5)] == "pre"
    assert phases[D(2024, 8, 9)] == "post"
    assert all(v is None for k, v in phases.items() if k not in (D(2024, 2, 5), D(2024, 8, 9)))


def test_rejected_periods_table_lists_every_rejection(mini):
    assert mini.execute("SELECT COUNT(*) FROM rejected_periods").fetchone() == (5,)
    assert mini.execute("SELECT COUNT(*) FROM accepted_periods").fetchone() == (2,)


def test_period_therms_sum_to_the_sum_of_read_deltas(mini):
    assert mini.execute("SELECT SUM(therms) FROM periods").fetchone() == (600.0,)


def test_therms_are_conserved_for_every_home(con):
    deltas = con.execute(
        """
        WITH ordered AS (
            SELECT
                home_id,
                reading - LAG(reading) OVER (PARTITION BY home_id ORDER BY read_date) AS d
            FROM meter_reads
        )
        SELECT home_id, SUM(d) AS total FROM ordered GROUP BY home_id
        """
    ).df()
    periods = con.execute("SELECT home_id, SUM(therms) AS total FROM periods GROUP BY home_id").df()
    merged = deltas.merge(periods, on="home_id", suffixes=("_reads", "_periods"))
    assert len(merged) == 600
    assert (merged["total_reads"] - merged["total_periods"]).abs().max() < 1e-6
    accepted_plus_rejected = con.execute(
        "SELECT (SELECT SUM(therms) FROM accepted_periods) "
        "+ (SELECT SUM(therms) FROM rejected_periods)"
    ).fetchone()[0]
    assert accepted_plus_rejected == pytest.approx(merged["total_reads"].sum(), abs=1e-3)


def test_no_accepted_period_touches_an_estimated_read(con):
    touching = con.execute(
        """
        SELECT COUNT(*) FROM accepted_periods p
        JOIN meter_reads m ON m.home_id = p.home_id AND m.is_estimated
            AND m.read_date IN (p.start_date, p.end_date)
        """
    ).fetchone()[0]
    assert touching == 0
    merged = con.execute("SELECT SUM(merged_estimated_reads) FROM periods").fetchone()[0]
    assert (
        merged == con.execute("SELECT COUNT(*) FROM meter_reads WHERE is_estimated").fetchone()[0]
    )


def test_meter_swap_homes_have_a_negative_period_that_never_enters_a_fit(con, synth_data):
    swap = synth_data.truth["meter_swap_homes"]
    negative = con.execute(
        "SELECT home_id, period_seq FROM rejected_periods "
        "WHERE reject_reason = 'negative_consumption'"
    ).fetchall()
    assert sorted(h for h, _ in negative) == sorted(swap)
    in_fit = con.execute(
        """
        SELECT COUNT(*) FROM period_hdd h
        JOIN rejected_periods r ON r.home_id = h.home_id AND r.period_seq = h.period_seq
        """
    ).fetchone()[0]
    assert in_fit == 0


def test_short_history_homes_fail_the_pre_phase_with_insufficient_periods(con, synth_data):
    short = synth_data.truth["short_history_homes"]
    rows = con.execute(
        "SELECT home_id, n, fail_reason FROM home_fits WHERE phase = 'pre' AND home_id IN "
        f"({','.join(map(str, short))})"
    ).fetchall()
    assert len(rows) == 12
    assert all(reason == "insufficient_periods" and n < 9 for _, n, reason in rows)


def test_no_period_overlapping_a_retrofit_month_is_in_a_phase(con):
    bad = con.execute(
        """
        SELECT COUNT(*) FROM accepted_periods
        WHERE phase IS NOT NULL
          AND start_date + 1 <= event_month_end AND end_date >= event_month_start
        """
    ).fetchone()[0]
    assert bad == 0
    pre_after = con.execute(
        "SELECT COUNT(*) FROM accepted_periods "
        "WHERE phase = 'pre' AND end_date >= event_month_start"
    ).fetchone()[0]
    assert pre_after == 0
    assert (
        con.execute(
            "SELECT COUNT(*) FROM rejected_periods WHERE reject_reason = 'blackout'"
        ).fetchone()[0]
        > 500
    )
