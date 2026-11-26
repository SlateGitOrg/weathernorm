-- Heating degree days at every candidate base temperature.
--
-- Inputs : weather(date, temp_f), accepted_periods
-- Outputs: candidate_bases, daily_hdd, period_hdd, normal_daily_hdd, normal_annual_hdd
--
-- HDD on a day is max(base - mean temperature, 0). A period's HDD is the sum over the days
-- (start_date, end_date], found with a date-range join. The normal year is the mean HDD by
-- calendar day over every year of weather, 29 February dropped.

CREATE OR REPLACE TABLE candidate_bases AS
SELECT range::INTEGER AS base FROM range(${base_min}, ${base_max} + 1);

CREATE OR REPLACE TABLE daily_hdd AS
SELECT
    w.date,
    b.base,
    GREATEST(b.base - w.temp_f, 0) AS hdd
FROM weather w
CROSS JOIN candidate_bases b
ORDER BY w.date, b.base;

CREATE OR REPLACE TABLE period_hdd AS
SELECT
    p.home_id,
    p.period_seq,
    d.base,
    SUM(d.hdd) AS hdd,
    COUNT(*) AS weather_days
FROM accepted_periods p
JOIN daily_hdd d
    ON d.date > p.start_date
    AND d.date <= p.end_date
WHERE p.phase IS NOT NULL
GROUP BY p.home_id, p.period_seq, d.base
ORDER BY p.home_id, p.period_seq, d.base;

CREATE OR REPLACE TABLE normal_daily_hdd AS
SELECT
    month(date) AS month,
    day(date) AS day,
    base,
    AVG(hdd) AS hdd
FROM daily_hdd
WHERE NOT (month(date) = 2 AND day(date) = 29)
GROUP BY month(date), day(date), base
ORDER BY base, month, day;

CREATE OR REPLACE TABLE normal_annual_hdd AS
SELECT base, SUM(hdd) AS hdd, COUNT(*) AS days
FROM normal_daily_hdd
GROUP BY base
ORDER BY base;
