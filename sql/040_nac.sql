-- Normalised annual consumption: what each home would use in a normal 365-day year.
--
-- Inputs : home_fits, normal_annual_hdd
-- Outputs: nac
--
-- NAC = 365 * alpha + beta * HDD_normal(base), using the home's chosen base and fit.

CREATE OR REPLACE TABLE nac AS
SELECT
    f.home_id,
    f.grp,
    f.phase,
    f.base,
    f.intercept AS alpha,
    f.slope AS beta,
    n.hdd AS normal_hdd,
    365 * f.intercept + f.slope * n.hdd AS nac
FROM home_fits f
JOIN normal_annual_hdd n USING (base)
WHERE f.fail_reason IS NULL
ORDER BY f.home_id, f.phase;
