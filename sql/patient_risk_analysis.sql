-- Operational analysis used to identify high-utilization patient segments.
SELECT
    patient_id,
    COUNT(*) AS encounters,
    SUM(CASE WHEN readmitted = 1 THEN 1 ELSE 0 END) AS readmissions,
    AVG(length_of_stay) AS avg_length_of_stay,
    ROUND(
        SUM(CASE WHEN readmitted = 1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0),
        4
    ) AS readmission_rate
FROM patient_encounters
GROUP BY patient_id
HAVING COUNT(*) >= 2
ORDER BY readmission_rate DESC, encounters DESC;
