# Postmortem: <short title>

*Blameless: describe systems and decisions, not people.*

| | |
|---|---|
| Date of incident | YYYY-MM-DD |
| Detected by | alert name / person / downstream user |
| Duration | from first bad data or failure to resolution |
| Severity | critical / warning |
| Author | |

## Summary

Two or three sentences: what broke, who or what was affected, how it was fixed.

## Impact

* Which data or dates were missing or wrong, and for how long.
* Which downstream results (alpha evaluations) used affected data. Were they re-run?

## Timeline (ICT)

| Time | Event |
|---|---|
| hh:mm | first symptom |
| hh:mm | alert fired / someone noticed |
| hh:mm | mitigation |
| hh:mm | resolved |

## Root cause

What actually happened, with evidence: log lines, `alphaops runs -v` output, the vendor
response. Ask "why" until you reach something you can change.

## What went well / what didn't

* Detection: did an alert catch it, or a person? How fast?
* Diagnosis: was the runbook right? What was missing?

## Action items

| Action | Type (prevent / detect / mitigate) | Owner | Ticket |
|---|---|---|---|
| e.g. add a check for X | detect | | |
