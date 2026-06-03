# Resume Project Entry

## Project Name

ADAS Scenario-Based Risk Mining & Regression Test Platform

## Role

Independent developer / test-platform engineer

## Tech Stack

Python, OpenCV, YOLOv8, pandas, NumPy, Streamlit, Matplotlib, pytest

## Resume Description

Built a local ADAS road-scene testing platform that analyzes driving videos and synthetic detection logs, identifies scenario-based risk events, generates evidence frames, and exports tester-style reports, defect records, and regression test cases.

## Bullet Points

- Developed a Python video-analysis pipeline using OpenCV and YOLOv8 to detect road users, classify frame-level risk events, save evidence frames, render annotated driving videos, and generate first-version ADAS closed-loop reports from real videos.
- Built a deterministic synthetic ADAS regression pipeline that mines close-following, front-vehicle deceleration, pedestrian-crossing, cut-in, and lane-departure-risk scenarios.
- Implemented TTC, THW, distance, relative-speed, and rule-based risk scoring to prioritize ADAS safety events.
- Automated defect triage and regression test-case generation with CSV, Excel, JSON, Markdown, HTML, chart, and Streamlit dashboard outputs.
- Added pytest coverage for risk rules, scenario mining, risk metrics, defect triage, report generation, and video rendering.

## Short Interview Line

This project demonstrates how to turn road-scene perception results into ADAS testing artifacts: risk events, evidence, defect reports, and regression test cases.
