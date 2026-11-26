-- Per home x phase x base regression of therms/day on HDD/day, and the balance-point choice.
--
-- Inputs : accepted_periods, period_hdd, homes
-- Outputs: fit_grid (every base), home_fits (one chosen row per home and phase)
--
-- Ordinary least squares by regr_slope / regr_intercept / regr_r2. The chosen base maximises R2;
-- ties go to the lower base. A chosen fit fails with `insufficient_periods` when it has fewer than
-- min_periods periods, else with `no_heating_signal` when its slope is not positive.

CREATE OR REPLACE TABLE fit_grid AS
SELECT
    p.home_id,
    p.phase,
    h.base,
    regr_slope(p.therms / p.days, h.hdd / p.days) AS slope,
    regr_intercept(p.therms / p.days, h.hdd / p.days) AS intercept,
    regr_r2(p.therms / p.days, h.hdd / p.days) AS r2,
    regr_count(p.therms / p.days, h.hdd / p.days) AS n
FROM accepted_periods p
JOIN period_hdd h USING (home_id, period_seq)
WHERE p.phase IS NOT NULL
GROUP BY p.home_id, p.phase, h.base
ORDER BY p.home_id, p.phase, h.base;

CREATE OR REPLACE TABLE home_fits AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY home_id, phase ORDER BY r2 DESC NULLS LAST, base ASC
        ) AS rk
    FROM fit_grid
),
slots AS (
    SELECT h.home_id, h.grp, ph.phase
    FROM homes h
    CROSS JOIN (VALUES ('pre'), ('post')) AS ph(phase)
)
SELECT
    s.home_id,
    s.grp,
    s.phase,
    r.base,
    r.slope,
    r.intercept,
    r.r2,
    COALESCE(r.n, 0) AS n,
    CASE
        WHEN COALESCE(r.n, 0) < ${min_periods} THEN 'insufficient_periods'
        WHEN r.slope IS NULL OR r.slope <= 0 THEN 'no_heating_signal'
    END AS fail_reason
FROM slots s
LEFT JOIN ranked r
    ON r.home_id = s.home_id AND r.phase = s.phase AND r.rk = 1
ORDER BY s.home_id, s.phase;
