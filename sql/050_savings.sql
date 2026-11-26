-- Savings: per home, per group, and the programme difference in differences.
--
-- Inputs : nac, accepted_periods
-- Outputs: home_savings, group_savings, naive_raw, programme_savings
--
-- Home savings = NAC_pre - NAC_post for homes with both phases fitted. Programme savings is the
-- participants' mean saving minus the comparison homes' mean saving, also shown as a percentage
-- of the participants' mean pre-phase NAC. The naive figure compares raw annualised bills in the
-- year before and the year after the retrofit month, with no weather adjustment.

CREATE OR REPLACE TABLE home_savings AS
SELECT
    pre.home_id,
    pre.grp,
    pre.base AS base_pre,
    post.base AS base_post,
    pre.nac AS nac_pre,
    post.nac AS nac_post,
    pre.nac - post.nac AS savings,
    100 * (pre.nac - post.nac) / pre.nac AS savings_pct
FROM nac pre
JOIN nac post ON post.home_id = pre.home_id AND post.phase = 'post'
WHERE pre.phase = 'pre'
ORDER BY pre.home_id;

CREATE OR REPLACE TABLE naive_raw AS
WITH windowed AS (
    SELECT
        home_id,
        grp,
        phase,
        SUM(therms) / SUM(days) * 365 AS annualised,
        COUNT(*) AS n_periods
    FROM accepted_periods
    WHERE in_year_window
    GROUP BY home_id, grp, phase
)
SELECT
    pre.home_id,
    pre.grp,
    pre.annualised AS raw_pre,
    post.annualised AS raw_post
FROM windowed pre
JOIN windowed post ON post.home_id = pre.home_id AND post.phase = 'post'
WHERE pre.phase = 'pre'
    AND pre.n_periods >= ${naive_min_periods}
    AND post.n_periods >= ${naive_min_periods}
ORDER BY pre.home_id;

CREATE OR REPLACE TABLE group_savings AS
WITH nac_part AS (
    SELECT
        grp,
        COUNT(*) AS n_homes,
        AVG(nac_pre) AS mean_nac_pre,
        AVG(nac_post) AS mean_nac_post,
        AVG(savings) AS mean_savings,
        STDDEV_SAMP(savings_pct) AS savings_pct_sd
    FROM home_savings
    GROUP BY grp
),
raw_part AS (
    SELECT
        grp,
        COUNT(*) AS n_raw_homes,
        AVG(raw_pre) AS mean_raw_pre,
        AVG(raw_post) AS mean_raw_post
    FROM naive_raw
    GROUP BY grp
)
SELECT
    n.grp,
    n.n_homes,
    n.mean_nac_pre,
    n.mean_nac_post,
    n.mean_savings,
    100 * n.mean_savings / n.mean_nac_pre AS nac_change_pct,
    n.savings_pct_sd,
    r.n_raw_homes,
    r.mean_raw_pre,
    r.mean_raw_post,
    100 * (r.mean_raw_pre - r.mean_raw_post) / r.mean_raw_pre AS raw_change_pct
FROM nac_part n
JOIN raw_part r USING (grp)
ORDER BY n.grp;

CREATE OR REPLACE TABLE programme_savings AS
SELECT
    p.n_homes AS n_participants,
    c.n_homes AS n_comparison,
    p.mean_savings AS participant_savings,
    c.mean_savings AS comparison_savings,
    p.mean_savings - c.mean_savings AS did_therms,
    100 * (p.mean_savings - c.mean_savings) / p.mean_nac_pre AS did_pct,
    p.raw_change_pct AS naive_pct,
    c.raw_change_pct AS comparison_raw_change_pct,
    p.nac_change_pct AS participant_nac_change_pct,
    c.nac_change_pct AS comparison_nac_change_pct,
    c.savings_pct_sd AS comparison_spread_pct
FROM group_savings p
JOIN group_savings c ON p.grp = 'participant' AND c.grp = 'comparison';
