# Student dashboard integration
The advanced student dashboard replaces static/dashboard.html, which the original FastAPI app serves. Existing DOM IDs and inline JavaScript are preserved. The previous dashboard is retained as static/dashboard.original.html. No API schema or database migration was performed. The original attendance policy and legacy employee_id/faculty database field names remain for compatibility; the UI wording alone does not convert staff attendance rules into a student timetable.

Run from the project root using the existing README and requirements. Configure authorized camera credentials and enrollment data locally. Optional SR/AdaFace weights and their compatible galleries must be supplied separately. Read TEST_REPORT_180.md before deployment.
