-- Practice queries. Tables: rfis, submittals (see sql/make_db.py)

-- 1. Plain SELECT / WHERE (this is the part that looks like SOQL)
SELECT rfi_number, subject FROM rfis WHERE status = 'Open';

-- 2. Counts by group (like GROUP BY in SOQL)
SELECT status, COUNT(*) AS how_many FROM submittals GROUP BY status;

-- 3. INNER JOIN: only rows that match in BOTH tables.
--    Open RFIs next to submittals still pending review in the same spec section.
SELECT r.rfi_number, r.subject, s.submittal_number, s.description
FROM rfis r
JOIN submittals s ON s.spec_section = r.spec_section
WHERE r.status = 'Open' AND s.status = 'Pending Review';

-- 4. LEFT JOIN: every row from the first table, blanks where nothing matches.
--    Which RFIs have no submittal linked to them?
SELECT r.rfi_number, r.subject
FROM rfis r
LEFT JOIN submittals s ON s.related_rfi = r.rfi_number
WHERE s.submittal_number IS NULL;

-- 5. Dates: average days to answer an RFI (SQLite uses julianday)
SELECT ROUND(AVG(julianday(date_answered) - julianday(date_submitted)), 1) AS avg_days_to_answer
FROM rfis
WHERE date_answered IS NOT NULL;

-- 6. JOIN + GROUP BY: RFIs that led to rejected or resubmitted submittals, by spec section
SELECT r.spec_section, COUNT(DISTINCT r.rfi_number) AS rfis, COUNT(s.submittal_number) AS problem_submittals
FROM rfis r
JOIN submittals s ON s.related_rfi = r.rfi_number
WHERE s.status IN ('Rejected', 'Revise and Resubmit')
GROUP BY r.spec_section;
