-- Billing periods from meter reads.
--
-- Inputs : meter_reads(home_id, read_date, reading, is_estimated), homes(home_id, grp, event_date)
-- Outputs: periods, accepted_periods, rejected_periods
--
-- An estimated read is not a measurement, so a period always runs from one actual read to the
-- next actual read; every estimated read in between is merged into it. A period covers the days
-- (start_date, end_date]. Each period gets at most one reject reason, tested in this order:
-- negative_consumption, too_short, too_long, blackout. Blackout periods overlap the calendar month
-- of the home's retrofit (or, for comparison homes, the same kind of reference date).

CREATE OR REPLACE TABLE actual_reads AS
SELECT
    home_id,
    read_date AS end_date,
    reading AS end_reading,
    LAG(read_date) OVER w AS start_date,
    LAG(reading) OVER w AS start_reading
FROM meter_reads
WHERE NOT is_estimated
WINDOW w AS (PARTITION BY home_id ORDER BY read_date);

CREATE OR REPLACE TABLE periods AS
WITH merged AS (
    SELECT
        a.home_id,
        a.start_date,
        a.end_date,
        date_diff('day', a.start_date, a.end_date) AS days,
        a.end_reading - a.start_reading AS therms,
        COUNT(m.read_date) AS merged_estimated_reads
    FROM actual_reads a
    LEFT JOIN meter_reads m
        ON m.home_id = a.home_id
        AND m.is_estimated
        AND m.read_date > a.start_date
        AND m.read_date < a.end_date
    WHERE a.start_date IS NOT NULL
    GROUP BY a.home_id, a.start_date, a.end_date, a.end_reading, a.start_reading
),
dated AS (
    SELECT
        m.*,
        h.grp,
        date_trunc('month', h.event_date)::DATE AS event_month_start,
        last_day(h.event_date) AS event_month_end
    FROM merged m
    JOIN homes h USING (home_id)
),
judged AS (
    SELECT
        *,
        CASE
            WHEN therms < 0 THEN 'negative_consumption'
            WHEN days < ${min_days} THEN 'too_short'
            WHEN days > ${max_days} THEN 'too_long'
            WHEN start_date < event_month_end AND end_date >= event_month_start THEN 'blackout'
        END AS reject_reason
    FROM dated
)
SELECT
    home_id,
    ROW_NUMBER() OVER (PARTITION BY home_id ORDER BY end_date) AS period_seq,
    grp,
    start_date,
    end_date,
    days,
    therms,
    merged_estimated_reads,
    reject_reason,
    CASE
        WHEN reject_reason IS NOT NULL THEN NULL
        WHEN end_date < event_month_start THEN 'pre'
        WHEN start_date >= event_month_end THEN 'post'
    END AS phase,
    -- One-year windows either side of the retrofit month, for the naive bill comparison only.
    CASE
        WHEN reject_reason IS NOT NULL THEN FALSE
        WHEN end_date < event_month_start
            THEN start_date >= event_month_start - INTERVAL 12 MONTH
        WHEN start_date >= event_month_end
            THEN end_date <= event_month_end + INTERVAL 12 MONTH
        ELSE FALSE
    END AS in_year_window,
    event_month_start,
    event_month_end
FROM judged
ORDER BY home_id, period_seq;

CREATE OR REPLACE TABLE accepted_periods AS
SELECT * FROM periods WHERE reject_reason IS NULL ORDER BY home_id, period_seq;

CREATE OR REPLACE TABLE rejected_periods AS
SELECT
    home_id, period_seq, start_date, end_date, days, therms, merged_estimated_reads, reject_reason
FROM periods
WHERE reject_reason IS NOT NULL
ORDER BY home_id, period_seq;
