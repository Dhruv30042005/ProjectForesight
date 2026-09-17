# FORESIGHT — Demand & Inventory Intelligence Platform

FORESIGHT is an AI-powered demand forecasting and inventory intelligence platform designed to support SKU-level inventory planning.

The system combines historical demand analysis, machine-learning-based forecasting, inventory risk scoring, estimated financial impact, and actionable inventory recommendations through an interactive dashboard and a deployed scoring API.

---

## 🚀 Live Demo

### Dashboard
https://dhruv30042005-projectforesight-app-got8p5.streamlit.app/

### Scoring API
https://foresight-scoring-api-hpkb.onrender.com

### API Documentation
https://foresight-scoring-api-hpkb.onrender.com/docs

### GitHub Repository
https://github.com/Dhruv30042005/ProjectForesight

---

## 🎯 Business Problem

Traditional inventory planning often relies on spreadsheets, historical intuition, and manual decision-making.

This can result in:

- Stockouts and lost sales
- Overstocked inventory
- Capital locked in excess stock
- Difficulty identifying high-risk products
- Limited visibility into future demand

FORESIGHT addresses these challenges by providing demand forecasts, inventory risk signals, financial impact estimates, and planning recommendations.

---

## 💡 Key Objectives

- Forecast weekly SKU-level demand
- Generate an 8-week future demand forecast
- Identify stockout risk
- Identify overstock risk
- Estimate potential financial impact in INR
- Classify inventory into decision quadrants
- Provide actionable inventory recommendations
- Provide an interactive planning dashboard
- Expose inventory scoring through a REST API

---

## 🏗️ System Architecture

```text
                Raw Retail Data
                       │
                       ▼
              Data Processing Pipeline
                       │
                       ▼
              Data Quality + EDA
                       │
                       ▼
              Demand Forecasting
                       │
                       ▼
             Inventory Risk Scoring
                 ┌─────┴─────┐
                 │           │
                 ▼           ▼
            Stockout      Overstock
              Risk           Risk
                 │           │
                 └─────┬─────┘
                       ▼
              Financial Impact
                       │
                       ▼
              Decision Quadrants
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Streamlit Dashboard    FastAPI
              │                 │
              ▼                 ▼
         Planning View      REST Scoring API