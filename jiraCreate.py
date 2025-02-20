import csv
import os

def create_jira_csv():
    # Your email for assignments
    your_email = "alec@apintegrations.com"
    
    # -------------------------------------------------------------------------
    # Enhanced sprint_plan: Epics/stories with references to Implementation Science,
    # NLP expansions, multi-level data approaches, and continuous feedback loops.
    # -------------------------------------------------------------------------
    sprint_plan = {
        "Epic 1: Ad Score Modeling Infrastructure": [
            {
                "summary": "Set up data lake environment with multi-level data approach",
                "description": (
                    "Based on Implementation Science principles, implement a scalable data lake "
                    "that supports multi-level data inputs (campaign-level, user-level, environment-level). "
                    "Focus on layering raw, processed, and curated data while maintaining robust security.\n\n"
                    "Acceptance Criteria:\n"
                    "- Data lake accommodates multiple data sources aligned with Implementation Science guidelines\n"
                    "- Security, governance, and compliance measures are in place\n"
                    "- Clear separation of raw vs processed vs curated layers\n"
                    "- References: 'Applying Machine Learning Techniques to Implementation Science' cheat sheet"
                )
            },
            {
                "summary": "Implement secure cross-team data governance with equity considerations",
                "description": (
                    "Develop and document data-sharing protocols that incorporate equity considerations "
                    "across different user demographics and contexts. Implement thorough access controls, encryption, "
                    "and auditing mechanisms.\n\n"
                    "Acceptance Criteria:\n"
                    "- Access controls enforced with role-based permissions\n"
                    "- Data is encrypted at rest and in transit\n"
                    "- Auditing logs track data usage, ensuring equitable access\n"
                    "- References: Implementation Science equity best practices"
                )
            },
            {
                "summary": "Create comprehensive data catalog with usage guidelines",
                "description": (
                    "Develop a data catalog to handle multi-level data structures and usage scenarios. "
                    "Document schema, lineage, usage, and ensure alignment with continuous feedback loops "
                    "for real-time updates.\n\n"
                    "Acceptance Criteria:\n"
                    "- Data catalog is easily searchable\n"
                    "- Metadata documented for each dataset, including context needed for each level\n"
                    "- Catalog supports continuous improvement cycles\n"
                    "- References: Implementation Science frameworks (feedback loops)"
                )
            },
            {
                "summary": "Develop preprocessing workflow for ad engagement data (NLP integration)",
                "description": (
                    "Create a preprocessing pipeline that normalizes raw ad data, including text fields "
                    "using advanced NLP steps (tokenization, sentiment analysis). Evaluate the cleaned "
                    "output with multi-level context to improve readiness for modeling.\n\n"
                    "Acceptance Criteria:\n"
                    "- Handling of missing values and outliers consistent with multi-level data\n"
                    "- NLP pipeline integrated for text fields (tokenization, sentiment)\n"
                    "- Processed data validated for accuracy\n"
                    "- References: 'Essential NLP Techniques for Data Analysis'"
                )
            },
            {
                "summary": "Create time-series feature extraction to capture seasonal trends",
                "description": (
                    "Build feature extraction logic for capturing rolling averages, seasonal variations, "
                    "and time-based patterns. The approach aligns with Implementation Science emphasis on "
                    "continuous monitoring.\n\n"
                    "Acceptance Criteria:\n"
                    "- Rolling averages, growth rates stored in a feature store\n"
                    "- Seasonal patterns tracked for each campaign or audience subgroup\n"
                    "- References: Implementation Science (monitoring, sustaining phase)"
                )
            },
            {
                "summary": "Implement advanced NLP pipeline for content analysis",
                "description": (
                    "Enhance the NLP pipeline to perform deeper sentiment analysis, topic modeling, and named entity "
                    "recognition for ad content. Include domain-specific handling for marketing terminology.\n\n"
                    "Acceptance Criteria:\n"
                    "- NLP pipeline processes text data with advanced techniques (topic modeling, NER)\n"
                    "- Sentiment model results integrated into ad scoring\n"
                    "- References: 'Essential NLP Techniques for Data Analysis'"
                )
            },
            {
                "summary": "Test multiple ML models for ad effectiveness",
                "description": (
                    "Compare performance of various ML models (e.g. random forest, XGBoost, neural networks) on predicting "
                    "ad effectiveness. Include multi-level features in the training set.\n\n"
                    "Acceptance Criteria:\n"
                    "- Evaluate each model with consistent metrics (precision, recall, F1)\n"
                    "- Incorporate feedback loops for continual improvement\n"
                    "- References: Implementation Science multi-level data approach"
                )
            },
            {
                "summary": "Design validation framework with feedback loops",
                "description": (
                    "Implement a validation framework to measure model performance using precision, recall, and F1, "
                    "while also collecting real-world feedback from key stakeholders. Reflect these insights back "
                    "into model updates.\n\n"
                    "Acceptance Criteria:\n"
                    "- Precision/recall/F1 computed at each iteration\n"
                    "- Stakeholder feedback channels integrated\n"
                    "- References: Implementation Science emphasis on adaptation in real-world contexts"
                )
            },
            {
                "summary": "Develop model registry for versioning and experiment tracking",
                "description": (
                    "Set up experiment tracking (hyperparams, dataset versions), letting the team compare "
                    "model versions over time. Align with Implementation Science's call for transparent, "
                    "iterative improvement.\n\n"
                    "Acceptance Criteria:\n"
                    "- Models in a versioned registry\n"
                    "- Experiment logs with data references\n"
                )
            },
            {
                "summary": "Create API endpoints for ad score inference (no performance data integration)",
                "description": (
                    "Develop RESTful APIs to serve real-time or near-real-time ad score predictions, "
                    "but without hooking into performance data streaming. Keep endpoint minimal.\n\n"
                    "Acceptance Criteria:\n"
                    "- API operational to accept ad feature requests\n"
                    "- Endpoints documented, tested\n"
                )
            },
            {
                "summary": "Implement dashboard for monitoring ad score trends",
                "description": (
                    "Set up a simple visualization dashboard to track ad score trends. Emphasize the Implementation "
                    "Science 'monitor, support, and sustain' principle, even though this dashboard is not integrated "
                    "with external data.\n\n"
                    "Acceptance Criteria:\n"
                    "- Dashboard displays key metrics and scores\n"
                    "- Refresh cycle documented for internal usage\n"
                )
            },
            {
                "summary": "Add optional retraining pipeline triggered by data drift (manual for now)",
                "description": (
                    "Establish a pipeline that can be manually triggered to retrain the ad score model on new data "
                    "if drift is detected. This aligns with Implementation Science’s adaptive strategies.\n\n"
                    "Acceptance Criteria:\n"
                    "- Retraining pipeline documented\n"
                    "- No real-time or automated triggers yet\n"
                )
            }
        ],
        "Epic 2: Ad Account Health Monitoring": [
            {
                "summary": "Set up account data ingest (no real-time integration)",
                "description": (
                    "Integrate an existing pipeline for collecting raw ad account data from multiple sources. "
                    "Stick to manual or batch ingestion.\n\n"
                    "Acceptance Criteria:\n"
                    "- Data sources connected\n"
                    "- Data ingestion verified\n"
                )
            },
            {
                "summary": "Transform raw account data into analysis-ready format",
                "description": (
                    "Develop a pipeline to transform the ingested account data into standardized tables, "
                    "including multi-level context (account-level, campaign-level). No real-time or streaming.\n\n"
                    "Acceptance Criteria:\n"
                    "- Transformation rules documented\n"
                    "- Data validated for completeness\n"
                    "- References: Implementation Science multi-level alignment"
                )
            },
            {
                "summary": "Create cross-platform schema for uniform analysis",
                "description": (
                    "Design a unified schema to store data from different ad platforms, ensuring consistent column "
                    "naming and data types. This helps handle multi-level differences (org, campaign, user groups).\n\n"
                    "Acceptance Criteria:\n"
                    "- Unified schema tested across multiple platforms\n"
                    "- Documented approach to handle schema changes\n"
                )
            },
            {
                "summary": "Develop composite health metrics with equity lens",
                "description": (
                    "Design new composite metrics to measure ad account health. Factor in equity/bias checks "
                    "where certain audiences might be underrepresented.\n\n"
                    "Acceptance Criteria:\n"
                    "- Metrics documented in a knowledge base\n"
                    "- Basic checks for underrepresented audience segments\n"
                )
            },
            {
                "summary": "Create baseline model for account performance forecasting",
                "description": (
                    "Build a simple baseline model (e.g. linear regression or shallow neural net) to forecast "
                    "ad account performance trends. No real-time or big orchestration needed.\n\n"
                    "Acceptance Criteria:\n"
                    "- Baseline model trained on historical data\n"
                    "- Documented approach for improvements"
                )
            },
            {
                "summary": "Implement anomaly detection for account-level metrics",
                "description": (
                    "Add an anomaly detection component (similar to the ad-level version) for early warnings. "
                    "Emphasize Implementation Science’s feedback loops to refine anomaly definitions.\n\n"
                    "Acceptance Criteria:\n"
                    "- Anomalies flagged with consistent threshold\n"
                    "- Warnings stored or logged\n"
                )
            },
            {
                "summary": "User feedback collection mechanism",
                "description": (
                    "Design a basic interface or form for account managers to provide feedback on anomalies "
                    "and model outputs. This fosters the iterative approach recommended by Implementation Science.\n\n"
                    "Acceptance Criteria:\n"
                    "- Feedback mechanism tested with internal team\n"
                    "- Feedback data archived for later analysis\n"
                )
            },
            {
                "summary": "Add optional A/B testing for model version improvements",
                "description": (
                    "Build a small-scale A/B testing framework to compare baseline vs. improved models on static data. "
                    "No real-time updates.\n\n"
                    "Acceptance Criteria:\n"
                    "- A/B methodology documented\n"
                    "- Reporting templates for results\n"
                )
            },
            {
                "summary": "Generate static performance reports for ad accounts",
                "description": (
                    "Create a system that outputs PDF or CSV performance reports with the new metrics "
                    "and anomalies flagged. No API or streaming.\n\n"
                    "Acceptance Criteria:\n"
                    "- Reports incorporate account health metrics\n"
                    "- Automatic generation is documented\n"
                )
            }
        ],
        "Epic 3: MLOps Infrastructure": [
            {
                "summary": "Implement basic automated testing for ML scripts",
                "description": (
                    "Write unit tests for the predictor scripts, focusing on data shape validation, error-handling, "
                    "and consistent results. Minimal approach—no real-time integration.\n\n"
                    "Acceptance Criteria:\n"
                    "- Unit tests cover all major functions\n"
                    "- Scripts fail gracefully on exceptions"
                )
            },
            {
                "summary": "Establish standard deployment steps (documentation)",
                "description": (
                    "Document a minimal process for packaging Python scripts, versioning them (git tags, for example), "
                    "and deploying to a dev or staging environment. No big CI/CD.\n\n"
                    "Acceptance Criteria:\n"
                    "- Steps clearly outlined in README\n"
                    "- Minimal friction for developers"
                )
            },
            {
                "summary": "Create model registry (manual updates)",
                "description": (
                    "Set up a directory or simple tool (like MLflow or a shared CSV) to track which model versions "
                    "are in use. Keep it simple and rely on manual updates.\n\n"
                    "Acceptance Criteria:\n"
                    "- Model versions tracked in the registry\n"
                    "- Team can see which model is currently used"
                )
            },
            {
                "summary": "Develop minimal performance monitoring for offline logs",
                "description": (
                    "Implement a script or small library that reads offline logs after the fact to gather metrics. "
                    "Focus on Implementation Science’s emphasis on continuous improvement, but do it manually.\n\n"
                    "Acceptance Criteria:\n"
                    "- Monitoring script can parse logs\n"
                    "- Metrics stored in a local database or CSV\n"
                )
            },
            {
                "summary": "Implement optional data drift script",
                "description": (
                    "Write a script that calculates data drift metrics between old training data and newly collected "
                    "samples. Trigger it manually once a week.\n\n"
                    "Acceptance Criteria:\n"
                    "- Drift calculation stored in a local file or DB\n"
                    "- Notification to re-train (manual) if drift is high"
                )
            },
            {
                "summary": "Basic alerting for model degradation via email",
                "description": (
                    "Send an email alert if offline metrics show a large drop in model performance. Emphasize the "
                    "Implementation Science principle of fast feedback loops.\n\n"
                    "Acceptance Criteria:\n"
                    "- Email triggers on performance drop\n"
                    "- Basic threshold-based approach"
                )
            },
            {
                "summary": "Prepare dev environment for local inference scaling",
                "description": (
                    "Set up a small dev environment allowing local or single-cloud instance scaling if needed. "
                    "No real-time auto-scaling.\n\n"
                    "Acceptance Criteria:\n"
                    "- Docs show how to spin up multiple instances\n"
                    "- Observed improvement in local throughput"
                )
            },
            {
                "summary": "Optional caching for frequent predictions (manual control)",
                "description": (
                    "Implement an in-process cache (e.g., a Python dictionary or Redis if available) to speed up "
                    "frequent predictions. Let the dev team turn caching on or off.\n\n"
                    "Acceptance Criteria:\n"
                    "- Cache reduces average prediction latency\n"
                    "- Clear instructions for enabling/disabling"
                )
            },
            {
                "summary": "Manual scaling instructions for peak times",
                "description": (
                    "Provide instructions on how to scale up the environment manually during promotional events "
                    "or known peak times. Keep things lightweight.\n\n"
                    "Acceptance Criteria:\n"
                    "- Step-by-step scaling doc created\n"
                    "- Dev team can scale the environment on short notice"
                )
            }
        ],
        "Epic 4: Knowledge Transfer & Documentation": [
            {
                "summary": "Document multi-level modeling approaches",
                "description": (
                    "Write up how each predictor script addresses multi-level data (ad-level, account-level, "
                    "context-level) and cite references from the Implementation Science cheat sheets.\n\n"
                    "Acceptance Criteria:\n"
                    "- Documentation explains multi-level approach\n"
                    "- Clear references to Implementation Science materials"
                )
            },
            {
                "summary": "Document NLP feature engineering",
                "description": (
                    "Explain all text-based features used or planned (sentiment, topic modeling, entity recognition), "
                    "with references to 'Essential NLP Techniques for Data Analysis.'\n\n"
                    "Acceptance Criteria:\n"
                    "- Feature list with short descriptions\n"
                    "- Link to external cheat sheet or references"
                )
            },
            {
                "summary": "Create troubleshooting guide for partial data or missing fields",
                "description": (
                    "Detail how to handle missing or incomplete data in a manner consistent with Implementation Science "
                    "(i.e., partial data can still provide partial insights). Provide step-by-step instructions.\n\n"
                    "Acceptance Criteria:\n"
                    "- Clear instructions for dealing with missing data\n"
                    "- References to real-world adaptation (Implementation Science)"
                )
            },
            {
                "summary": "Onboarding materials emphasizing feedback loops",
                "description": (
                    "Develop onboarding docs that highlight the importance of collecting real-world feedback from "
                    "teams and adjusting the ML pipeline accordingly, as recommended by Implementation Science.\n\n"
                    "Acceptance Criteria:\n"
                    "- Onboarding doc includes rationale for feedback loops\n"
                    "- Real examples from pilot usage"
                )
            },
            {
                "summary": "User guides for non-technical stakeholders (focusing on new NLP features)",
                "description": (
                    "Write simple guides that show non-technical stakeholders how to interpret ad score or account "
                    "health predictions, emphasizing the NLP-derived insights.\n\n"
                    "Acceptance Criteria:\n"
                    "- Guides emphasize interpretability of text-based features\n"
                    "- Glossary of NLP terms provided"
                )
            },
            {
                "summary": "Implement minimal model explainability visuals",
                "description": (
                    "Create basic charts or diagrams (e.g., SHAP summary) to illustrate how features affect predictions. "
                    "Encourage further improvements in the future.\n\n"
                    "Acceptance Criteria:\n"
                    "- Visuals exist for each model\n"
                    "- Basic interpretability for stakeholders"
                )
            }
        ],
        "Epic 5: Risk Management & Mitigation": [
            {
                "summary": "Implement data validation checks focusing on contextual integrity",
                "description": (
                    "Add checks that ensure data is valid in the multi-level sense, so that account-level metrics "
                    "never override ad-level data. Outline how you handle potential biases.\n\n"
                    "Acceptance Criteria:\n"
                    "- Validation scripts exist\n"
                    "- Potential biases flagged (e.g., missing subgroups)\n"
                )
            },
            {
                "summary": "Create minimal synthetic data generator for testing",
                "description": (
                    "Develop a script that produces dummy data for each level (ad, account, environment). "
                    "Use it to test pipeline logic and catch data processing issues.\n\n"
                    "Acceptance Criteria:\n"
                    "- Synthetic data covers multiple ad contexts\n"
                    "- Pipeline runs without errors"
                )
            },
            {
                "summary": "Establish realistic benchmarks referencing Implementation Science results",
                "description": (
                    "Define performance benchmarks (e.g. baseline accuracy, recall) that reference known results "
                    "from Implementation Science and the NLP cheat sheet pilots. Make them realistic.\n\n"
                    "Acceptance Criteria:\n"
                    "- Benchmarks published in docs\n"
                    "- Team agreement on near-term vs. long-term goals"
                )
            },
            {
                "summary": "Develop hybrid approach with fallback rules for ad scoring",
                "description": (
                    "Ensure that if ML predictions fail or are uncertain, simple rule-based logic catches obvious outliers. "
                    "Align with Implementation Science's call for practical solutions in resource-limited settings.\n\n"
                    "Acceptance Criteria:\n"
                    "- Hybrid fallback implemented in ad score scripts\n"
                    "- Logging differentiates rule-based from ML-based output"
                )
            },
            {
                "summary": "Document phased rollout plan for new ML features",
                "description": (
                    "Create a basic plan for rolling out new or updated models in stages (e.g., internal test, limited user test, "
                    "full deployment), referencing Implementation Science's iterative approach.\n\n"
                    "Acceptance Criteria:\n"
                    "- Rollout phases clearly outlined\n"
                    "- Team alignment on testing scope"
                )
            },
            {
                "summary": "Schedule recurring knowledge-sharing sessions",
                "description": (
                    "Institute a monthly session where developers, data scientists, and stakeholders review model outcomes, "
                    "NLP improvements, and Implementation Science insights.\n\n"
                    "Acceptance Criteria:\n"
                    "- Sessions on the calendar\n"
                    "- Meeting notes shared in a central doc"
                )
            }
        ]
    }

    # Create CSV file
    csv_file = "within_ml_implementation.csv"
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row
        writer.writerow([
            'Summary', 'Issue Type', 'Description', 'Priority',
            'Labels', 'Epic Link', 'Issue ID', 'Parent', 'Assignee', 'Reporter'
        ])
        
        epic_count = 0
        story_count = 0
        
        for epic, stories in sprint_plan.items():
            epic_count += 1
            epic_id = f"EPIC-{epic_count}"
            
            # Add the epic
            writer.writerow([
                epic,  # Summary
                'Epic',  
                (f"Machine learning implementation for WITHIN's ad scoring "
                 f"and account health systems, integrating Implementation Science "
                 f"and NLP best practices.\n\n{epic}"),  # Enhanced epic description
                'Medium',
                'WITHIN,ML,MachineLearning',
                '',  # Epic Link
                epic_id,
                '',  # Parent
                your_email,
                your_email
            ])
            
            # Add the stories for this epic
            for story in stories:
                story_count += 1
                story_id = f"STORY-{story_count}"
                
                writer.writerow([
                    story["summary"],
                    'Story',
                    story["description"],
                    'Medium',
                    'WITHIN,ML',  
                    epic,
                    story_id,
                    epic_id,
                    your_email,
                    your_email
                ])
    
    print(f"CSV file created: {os.path.abspath(csv_file)}")
    print(f"All tasks assigned to: {your_email}")
    print("\nInstructions for importing into Jira:")
    print("1. Log into your Jira Cloud instance")
    print("2. Choose  > System")
    print("3. Under 'Import and Export', click 'External System Import'")
    print("4. Click 'CSV'")
    print("5. Upload your CSV file and follow the wizard")
    print("6. Map the Assignee and Reporter fields during the import")


if __name__ == "__main__":
    create_jira_csv()
