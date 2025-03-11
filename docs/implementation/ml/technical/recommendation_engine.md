# Recommendation Engine for Account Health Optimization

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document details the recommendation engine component of the Account Health Predictor, which is responsible for generating actionable optimization suggestions based on account health assessments and identified issues.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Recommendation Generation Workflow](#recommendation-generation-workflow)
4. [Recommendation Algorithms](#recommendation-algorithms)
5. [Recommendation Categories](#recommendation-categories)
6. [Implementation Details](#implementation-details)
7. [Prioritization Framework](#prioritization-framework)
8. [Impact Assessment](#impact-assessment)
9. [Explainability](#explainability)
10. [Recommendation API](#recommendation-api)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Future Improvements](#future-improvements)

## Introduction

The recommendation engine is a critical component of the Account Health Predictor system that transforms diagnostic findings into actionable recommendations for advertisers. By analyzing account health metrics, performance trends, and identified issues, the engine generates specific, prioritized recommendations designed to optimize account performance and address potential problems.

### Objectives

The primary objectives of the recommendation engine are to:

1. Generate relevant, actionable recommendations based on account health assessments
2. Prioritize recommendations for maximum impact
3. Provide clear explanations of expected outcomes
4. Track recommendation effectiveness over time
5. Adapt to changing industry conditions and platform updates

### Design Philosophy

The recommendation engine follows these key design principles:

1. **Contextual Awareness**: Recommendations consider the account's industry, size, objectives, and historical performance.
2. **Multi-source Knowledge**: Combines rule-based expertise with pattern-based machine learning.
3. **Explainability**: Each recommendation includes clear rationale and expected outcomes.
4. **Feedback Integration**: Continuously learns from the effectiveness of past recommendations.
5. **Platform-specific Optimization**: Tailors recommendations to the specific advertising platforms used.

## System Architecture

The recommendation engine employs a hybrid architecture combining rule-based systems with machine learning models:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                      ACCOUNT HEALTH PREDICTOR                       │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                        RECOMMENDATION ENGINE                        │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   Rule-Based    │  │ Machine Learning│  │ Historical Pattern  │  │
│  │  Recommender    │◄─┤   Recommender   │◄─┤     Analysis        │  │
│  └─────────┬───────┘  └─────────┬───────┘  └─────────────────────┘  │
│            │                    │                                    │
│            └──────────┬─────────┘                                    │
│                       │                                              │
│                       ▼                                              │
│  ┌────────────────────────────────────────┐ ┌──────────────────────┐│
│  │      Recommendation Aggregator         │ │    Optimization      ││
│  │      and Conflict Resolver             │ │    Database          ││
│  └────────────────────┬───────────────────┘ └──────────────────────┘│
│                       │                                              │
│                       ▼                                              │
│  ┌────────────────────────────────────────┐ ┌──────────────────────┐│
│  │      Recommendation Prioritizer        │ │    Impact            ││
│  │                                        │ │    Estimator         ││
│  └────────────────────┬───────────────────┘ └──────────────────────┘│
│                       │                                              │
│                       ▼                                              │
│  ┌────────────────────────────────────────┐ ┌──────────────────────┐│
│  │      Recommendation Formatter          │ │    Explanation       ││
│  │      and Delivery                      │ │    Generator         ││
│  └────────────────────┬───────────────────┘ └──────────────────────┘│
│                       │                                              │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                   ACTIONABLE RECOMMENDATIONS                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Rule-Based Recommender**: Applies expert-defined rules to generate recommendations based on account conditions.
2. **Machine Learning Recommender**: Uses supervised learning to suggest optimizations based on patterns in successful accounts.
3. **Historical Pattern Analysis**: Identifies effective past optimizations for similar accounts and conditions.
4. **Recommendation Aggregator**: Combines and deduplicates recommendations from multiple sources.
5. **Conflict Resolver**: Identifies and resolves contradictory recommendations.
6. **Recommendation Prioritizer**: Orders recommendations by expected impact and implementation effort.
7. **Impact Estimator**: Predicts the quantitative impact of each recommendation.
8. **Explanation Generator**: Creates human-readable explanations for each recommendation.
9. **Recommendation Formatter**: Standardizes recommendation format for delivery.

## Recommendation Generation Workflow

The recommendation generation process follows this workflow:

1. **Issue Detection**
   - Account health assessment identifies potential issues
   - Performance metrics are analyzed for anomalies and opportunities
   - Account structure is evaluated for optimization potential

2. **Context Gathering**
   - Account historical performance is analyzed
   - Industry benchmarks are retrieved
   - Campaign objectives and constraints are considered
   - Platform-specific capabilities are factored in

3. **Candidate Generation**
   - Rule-based system generates initial recommendations
   - ML model suggests optimizations based on similar accounts
   - Historical pattern analysis identifies previously successful tactics

4. **Filtering and Deduplication**
   - Redundant recommendations are combined
   - Contradictory recommendations are resolved
   - Context-inappropriate suggestions are filtered out

5. **Prioritization**
   - Recommendations are scored by expected impact
   - Implementation difficulty is assessed
   - Urgency based on issue severity is factored in
   - Final priority score is calculated

6. **Impact Estimation**
   - Expected performance improvements are quantified
   - Confidence intervals are calculated
   - Time-to-impact is estimated

7. **Explanation Generation**
   - Clear rationale is provided for each recommendation
   - Expected outcomes are explained
   - Implementation guidance is included

## Recommendation Algorithms

### Rule-Based System

The rule-based recommender employs a decision tree architecture with expert-defined rules that map specific account conditions to recommended actions:

```python
def rule_based_recommender(account_data, health_assessment, config):
    """
    Generate recommendations using rule-based approach.
    
    Args:
        account_data: DataFrame with account metrics and structure
        health_assessment: Results from Account Health Predictor
        config: Configuration for the rule engine
        
    Returns:
        List of recommendation objects
    """
    recommendations = []
    
    # Apply bidding optimization rules
    if "bidding_issues" in health_assessment:
        for issue in health_assessment["bidding_issues"]:
            if issue["type"] == "high_cpc":
                recommendations.append({
                    "category": "bidding",
                    "action": "reduce_bids",
                    "target": issue["campaign_id"],
                    "recommended_value": issue["current_bid"] * 0.9,
                    "rationale": "CPC is 30% above industry benchmark",
                    "expected_impact": {
                        "cpc_reduction": "-10-15%",
                        "conversion_impact": "0 to -5%"
                    },
                    "priority_score": calculate_priority_score(issue, "high_cpc")
                })
    
    # Apply budget allocation rules
    if "budget_issues" in health_assessment:
        for issue in health_assessment["budget_issues"]:
            if issue["type"] == "limited_by_budget" and issue["performance_score"] > 8:
                recommendations.append({
                    "category": "budget",
                    "action": "increase_budget",
                    "target": issue["campaign_id"],
                    "recommended_value": issue["current_budget"] * 1.2,
                    "rationale": "Campaign is limited by budget and performing well",
                    "expected_impact": {
                        "impression_increase": "+15-25%",
                        "conversion_increase": "+15-20%"
                    },
                    "priority_score": calculate_priority_score(issue, "limited_by_budget")
                })
    
    # Apply keyword optimization rules
    if "keyword_issues" in health_assessment:
        # Implementation of keyword recommendation rules
        pass
        
    # Apply creative optimization rules
    if "creative_issues" in health_assessment:
        # Implementation of creative recommendation rules
        pass
    
    # Apply targeting optimization rules
    if "targeting_issues" in health_assessment:
        # Implementation of targeting recommendation rules
        pass
    
    # Apply account structure optimization rules
    if "structure_issues" in health_assessment:
        # Implementation of structure recommendation rules
        pass
        
    return recommendations
```

### Machine Learning Model

The ML-based recommender uses a gradient boosting model trained on historical optimization actions and their outcomes:

```python
def ml_recommender(account_data, health_assessment, model):
    """
    Generate recommendations using machine learning.
    
    Args:
        account_data: DataFrame with account metrics and structure
        health_assessment: Results from Account Health Predictor
        model: Trained recommendation model
        
    Returns:
        List of recommendation objects
    """
    # Extract features for recommendation generation
    features = extract_recommendation_features(account_data, health_assessment)
    
    # Generate recommendation candidates
    candidates = generate_recommendation_candidates(account_data)
    
    recommendations = []
    
    # Score each candidate recommendation
    for candidate in candidates:
        # Combine candidate features with account features
        candidate_features = combine_features(features, candidate)
        
        # Predict impact score
        impact_score = model.predict_impact(candidate_features)
        
        # Predict success probability
        success_probability = model.predict_success_probability(candidate_features)
        
        if success_probability > 0.7:  # Only include high-confidence recommendations
            recommendations.append({
                "category": candidate["category"],
                "action": candidate["action"],
                "target": candidate["target"],
                "recommended_value": candidate["value"],
                "rationale": generate_ml_rationale(candidate, impact_score),
                "expected_impact": predict_impact_metrics(model, candidate_features),
                "priority_score": impact_score * success_probability,
                "confidence": float(success_probability)
            })
    
    return recommendations
```

### Historical Pattern Analysis

The historical pattern analyzer identifies patterns from successful past optimizations:

```python
def historical_pattern_recommender(account_data, similar_accounts, optimization_history):
    """
    Generate recommendations based on historical patterns.
    
    Args:
        account_data: DataFrame with account metrics and structure
        similar_accounts: List of similar accounts
        optimization_history: DataFrame with historical optimizations
        
    Returns:
        List of recommendation objects
    """
    recommendations = []
    
    # Find similar accounts
    account_clusters = find_account_clusters(account_data, similar_accounts)
    
    # Identify successful optimizations in similar accounts
    successful_optimizations = filter_successful_optimizations(
        optimization_history, 
        account_clusters
    )
    
    # Group by optimization type
    for optimization_type, optimizations in group_optimizations(successful_optimizations):
        # Calculate average impact
        avg_impact = calculate_average_impact(optimizations)
        
        if avg_impact["lift"] > 0.1:  # Only include impactful optimizations
            # Find best parameters
            best_params = find_best_parameters(optimizations)
            
            # Create recommendation
            recommendations.append({
                "category": optimization_type["category"],
                "action": optimization_type["action"],
                "target": identify_target(account_data, optimization_type),
                "recommended_value": best_params,
                "rationale": "This optimization improved performance by {:.1f}% in similar accounts".format(
                    avg_impact["lift"] * 100
                ),
                "expected_impact": translate_impact_metrics(avg_impact),
                "priority_score": calculate_historical_priority(avg_impact),
                "confidence": calculate_confidence(optimizations)
            })
    
    return recommendations
```

## Recommendation Categories

The recommendation engine generates suggestions across these primary categories:

### 1. Bidding Optimizations

Recommendations related to bid adjustments and bidding strategies:

| Type | Description | Examples |
|------|-------------|----------|
| Bid Adjustments | Changes to bid amounts | "Increase bids for top-performing keywords by 20%" |
| Bidding Strategy | Changes to bidding approach | "Switch to Target CPA bidding for Campaign X" |
| Device Modifiers | Bid adjustments by device | "Increase mobile bids by 15% on weekends" |
| Audience Modifiers | Bid adjustments by audience | "Increase bids for in-market segments by 10%" |

### 2. Budget Allocations

Recommendations for budget changes and redistributions:

| Type | Description | Examples |
|------|-------------|----------|
| Budget Increases | Suggestions to increase budgets | "Increase budget for Campaign X by $500/day" |
| Budget Redistribution | Reallocation of existing budget | "Shift 30% of budget from Campaign Y to Campaign X" |
| Dayparting Adjustments | Time-based budget allocation | "Allocate 40% of budget to evening hours (6-10pm)" |
| Seasonal Adjustments | Seasonal budget changes | "Increase holiday campaign budget by 50% in December" |

### 3. Creative Optimizations

Recommendations for ad creative improvements:

| Type | Description | Examples |
|------|-------------|----------|
| Ad Rotation | Changes to ad serving | "Test 3 new ad variations against control" |
| Ad Copy Suggestions | Improvements to ad text | "Include more specific benefits in headline" |
| Creative Elements | Addition/removal of elements | "Add price points to expanded text ads" |
| Asset Optimization | Improvements to visual assets | "Replace underperforming images in responsive ads" |

### 4. Keyword Optimizations

Recommendations for keyword adjustments:

| Type | Description | Examples |
|------|-------------|----------|
| Keyword Additions | New keywords to add | "Add 15 high-potential keywords from research" |
| Keyword Removals | Keywords to pause or remove | "Pause 7 low-performing, high-cost keywords" |
| Match Type Changes | Adjustments to match types | "Convert broad match keywords to phrase match" |
| Negative Keywords | Addition of negative keywords | "Add 12 negative keywords to reduce wasted spend" |

### 5. Targeting Optimizations

Recommendations for audience and targeting adjustments:

| Type | Description | Examples |
|------|-------------|----------|
| Audience Expansion | New audiences to target | "Add in-market segments for related products" |
| Audience Exclusions | Audiences to exclude | "Exclude converted customers from acquisition campaigns" |
| Geographic Targeting | Location-based adjustments | "Expand targeting to 5 additional high-performance cities" |
| Placement Optimizations | Site placement changes | "Exclude 12 underperforming placements" |

### 6. Account Structure Optimizations

Recommendations for campaign and account organization:

| Type | Description | Examples |
|------|-------------|----------|
| Campaign Segmentation | Structural changes to campaigns | "Split Campaign X into 3 targeted campaigns" |
| Ad Group Restructuring | Reorganization of ad groups | "Consolidate 5 similar ad groups" |
| Quality Score Improvements | Structure changes for quality | "Create dedicated landing pages for top keywords" |
| Campaign Settings | Changes to campaign settings | "Enable accelerated delivery for seasonal campaign" |

## Implementation Details

### Recommendation Object Schema

Each recommendation uses this standardized schema:

```json
{
  "recommendation_id": "rec_12345",
  "timestamp": "2025-03-15T14:30:00Z",
  "account_id": "act_67890",
  "category": "bidding",
  "action": "increase_bids",
  "target": {
    "type": "keyword",
    "id": "kw_123456",
    "name": "premium widgets",
    "campaign_id": "camp_78901",
    "campaign_name": "Widget Campaign"
  },
  "current_value": 1.25,
  "recommended_value": 1.50,
  "change_percentage": 20,
  "rationale": "This keyword shows strong conversion performance but low impression share due to rank.",
  "expected_impact": {
    "primary": {
      "metric": "impressions",
      "change": "+15-25%",
      "confidence": 0.85
    },
    "secondary": {
      "metric": "conversions",
      "change": "+10-18%",
      "confidence": 0.75
    },
    "negative": {
      "metric": "cpa",
      "change": "+3-8%",
      "confidence": 0.65
    }
  },
  "implementation_effort": "low",
  "urgency": "medium",
  "priority_score": 78,
  "source": "ml_recommender",
  "confidence": 0.82,
  "implementation_guide": "Navigate to Keywords > bid_strategy_01 > premium widgets and update bid to $1.50"
}
```

### Rule Engine Implementation

The rule engine is implemented using a flexible, configurable system:

```python
class RuleEngine:
    """Rule engine for generating recommendations."""
    
    def __init__(self, rule_config_path):
        """
        Initialize the rule engine with configuration.
        
        Args:
            rule_config_path: Path to rule configuration file
        """
        self.rules = self._load_rules(rule_config_path)
        self.context = {}
    
    def _load_rules(self, config_path):
        """Load rules from configuration file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def set_context(self, context):
        """
        Set context data for rule evaluation.
        
        Args:
            context: Dictionary with context data
        """
        self.context = context
    
    def evaluate_condition(self, condition):
        """
        Evaluate a rule condition against current context.
        
        Args:
            condition: Condition to evaluate
            
        Returns:
            Boolean indicating if condition is met
        """
        # Extract condition components
        metric = condition.get("metric")
        operator = condition.get("operator")
        value = condition.get("value")
        
        # Get actual metric value from context
        actual_value = self._get_metric_value(metric)
        
        # Evaluate condition
        if operator == "greater_than":
            return actual_value > value
        elif operator == "less_than":
            return actual_value < value
        elif operator == "equal_to":
            return actual_value == value
        elif operator == "contains":
            return value in actual_value
        elif operator == "percentage_change":
            baseline = self._get_metric_baseline(metric)
            if baseline == 0:
                return False
            change = (actual_value - baseline) / baseline
            return self._evaluate_percentage_condition(change, operator, value)
        
        return False
    
    def generate_recommendations(self):
        """
        Generate recommendations based on rules and context.
        
        Returns:
            List of recommendation objects
        """
        recommendations = []
        
        for rule in self.rules:
            # Check if all conditions are met
            conditions_met = all(
                self.evaluate_condition(condition)
                for condition in rule.get("conditions", [])
            )
            
            if conditions_met:
                # Generate recommendation from rule
                recommendation = self._create_recommendation_from_rule(rule)
                recommendations.append(recommendation)
        
        return recommendations
    
    def _create_recommendation_from_rule(self, rule):
        """Create recommendation object from matched rule."""
        # Implementation details for creating recommendation from rule
        
        # Format actual values into the recommendation
        target = self._resolve_target(rule.get("target"))
        current_value = self._get_current_value(target)
        recommended_value = self._calculate_recommended_value(rule, current_value)
        
        return {
            "recommendation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "account_id": self.context.get("account_id"),
            "category": rule.get("category"),
            "action": rule.get("action"),
            "target": target,
            "current_value": current_value,
            "recommended_value": recommended_value,
            "change_percentage": self._calculate_change_percentage(current_value, recommended_value),
            "rationale": self._format_rationale(rule.get("rationale_template")),
            "expected_impact": self._calculate_expected_impact(rule, current_value, recommended_value),
            "implementation_effort": rule.get("implementation_effort", "medium"),
            "urgency": rule.get("urgency", "medium"),
            "priority_score": rule.get("base_priority", 50),
            "source": "rule_engine",
            "confidence": rule.get("confidence", 0.7),
            "implementation_guide": self._format_implementation_guide(rule.get("implementation_template"))
        }
```

### ML Model Architecture

The recommendation ML model uses a gradient boosting architecture:

```python
def build_recommendation_model(training_data, config):
    """
    Build and train the recommendation ML model.
    
    Args:
        training_data: DataFrame with historical recommendations and outcomes
        config: Model configuration parameters
        
    Returns:
        Trained model
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    
    # Prepare features and targets
    X = training_data[config["feature_columns"]]
    y_impact = training_data[config["impact_column"]]
    y_success = training_data[config["success_column"]]
    
    # Split data
    X_train, X_test, y_impact_train, y_impact_test, y_success_train, y_success_test = train_test_split(
        X, y_impact, y_success, test_size=0.2, random_state=42
    )
    
    # Train impact prediction model
    impact_model = lgb.LGBMRegressor(
        n_estimators=config.get("n_estimators", 100),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 8),
        num_leaves=config.get("num_leaves", 31),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=42
    )
    
    impact_model.fit(
        X_train, 
        y_impact_train,
        eval_set=[(X_test, y_impact_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Train success prediction model
    success_model = lgb.LGBMClassifier(
        n_estimators=config.get("n_estimators", 100),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 8),
        num_leaves=config.get("num_leaves", 31),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=42
    )
    
    success_model.fit(
        X_train, 
        y_success_train,
        eval_set=[(X_test, y_success_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return {
        "impact_model": impact_model,
        "success_model": success_model,
        "feature_names": config["feature_columns"]
    }
```

## Prioritization Framework

Recommendations are prioritized using a scoring system that considers multiple factors:

### Priority Score Calculation

```python
def calculate_priority_score(recommendation, account_context):
    """
    Calculate priority score for a recommendation.
    
    Args:
        recommendation: Recommendation object
        account_context: Dictionary with account context
        
    Returns:
        Integer priority score (0-100)
    """
    # Base scores by category
    category_base_scores = {
        "bidding": 60,
        "budget": 70,
        "creative": 50,
        "keyword": 65,
        "targeting": 55,
        "structure": 45
    }
    
    # Base score by category
    base_score = category_base_scores.get(recommendation["category"], 50)
    
    # Impact score (0-40)
    impact_score = calculate_impact_score(recommendation["expected_impact"])
    
    # Effort score (0-20, higher for lower effort)
    effort_multipliers = {
        "very_low": 1.0,
        "low": 0.9,
        "medium": 0.7,
        "high": 0.5,
        "very_high": 0.3
    }
    effort_score = 20 * effort_multipliers.get(recommendation["implementation_effort"], 0.7)
    
    # Urgency score (0-20)
    urgency_multipliers = {
        "very_high": 1.0,
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
        "very_low": 0.2
    }
    urgency_score = 20 * urgency_multipliers.get(recommendation["urgency"], 0.6)
    
    # Account context adjustments
    account_adjustments = 0
    
    # If recommendation addresses primary optimization goal
    if recommendation["category"] == account_context.get("primary_goal", ""):
        account_adjustments += 10
    
    # If recommendation targets high-spend area
    if is_high_spend_target(recommendation["target"], account_context):
        account_adjustments += 5
    
    # Calculate final score
    priority_score = (base_score * 0.2) + impact_score + effort_score + urgency_score + account_adjustments
    
    # Ensure score is within 0-100 range
    priority_score = max(0, min(100, priority_score))
    
    return round(priority_score)
```

### Impact Estimation

The recommendation engine estimates the impact of each recommendation:

```python
def estimate_recommendation_impact(recommendation, account_data, model):
    """
    Estimate the impact of a recommendation.
    
    Args:
        recommendation: Recommendation object
        account_data: DataFrame with account metrics
        model: Impact prediction model
        
    Returns:
        Dictionary with impact estimates
    """
    # Extract features for impact prediction
    features = extract_impact_features(recommendation, account_data)
    
    # Predict primary impact
    primary_impact = model.predict_impact(features)
    
    # Determine impact metric based on recommendation category
    impact_metrics = determine_impact_metrics(recommendation["category"], recommendation["action"])
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(model, features)
    
    # Format impact estimates
    impact = {
        "primary": {
            "metric": impact_metrics["primary"],
            "change": format_percentage_range(
                confidence_intervals["lower"],
                confidence_intervals["upper"]
            ),
            "confidence": model.predict_confidence(features)
        }
    }
    
    # Add secondary impacts if applicable
    if "secondary" in impact_metrics:
        secondary_impact = estimate_secondary_impact(
            recommendation, 
            impact_metrics["secondary"],
            primary_impact
        )
        
        impact["secondary"] = {
            "metric": impact_metrics["secondary"],
            "change": format_percentage_range(
                secondary_impact["lower"],
                secondary_impact["upper"]
            ),
            "confidence": secondary_impact["confidence"]
        }
    
    # Add potential negative impacts if applicable
    if "negative" in impact_metrics:
        negative_impact = estimate_negative_impact(
            recommendation, 
            impact_metrics["negative"],
            primary_impact
        )
        
        impact["negative"] = {
            "metric": impact_metrics["negative"],
            "change": format_percentage_range(
                negative_impact["lower"],
                negative_impact["upper"]
            ),
            "confidence": negative_impact["confidence"]
        }
    
    return impact
```

## Explainability

The recommendation engine provides detailed explanations for each suggestion:

### Rationale Generation

```python
def generate_recommendation_rationale(recommendation, account_data, context):
    """
    Generate human-readable rationale for a recommendation.
    
    Args:
        recommendation: Recommendation object
        account_data: DataFrame with account metrics
        context: Additional context information
        
    Returns:
        String rationale
    """
    # Get template based on recommendation type
    template = get_rationale_template(
        recommendation["category"], 
        recommendation["action"]
    )
    
    # Gather metrics for context
    metrics = {
        "current_value": recommendation["current_value"],
        "recommended_value": recommendation["recommended_value"],
        "change_percentage": recommendation["change_percentage"]
    }
    
    # Add performance metrics if available
    if "target" in recommendation and "id" in recommendation["target"]:
        target_id = recommendation["target"]["id"]
        target_type = recommendation["target"]["type"]
        
        performance_metrics = get_performance_metrics(
            account_data, target_type, target_id
        )
        
        metrics.update(performance_metrics)
    
    # Add benchmark comparisons if available
    if "benchmarks" in context:
        benchmark_metrics = get_benchmark_comparisons(
            recommendation, metrics, context["benchmarks"]
        )
        
        metrics.update(benchmark_metrics)
    
    # Format the rationale with metrics
    rationale = template.format(**metrics)
    
    return rationale
```

### Implementation Guide Generation

```python
def generate_implementation_guide(recommendation, platform):
    """
    Generate step-by-step implementation instructions.
    
    Args:
        recommendation: Recommendation object
        platform: Advertising platform (e.g., "google", "facebook")
        
    Returns:
        String implementation guide
    """
    # Get platform-specific template
    template = get_implementation_template(
        platform,
        recommendation["category"],
        recommendation["action"]
    )
    
    # Gather path information
    target = recommendation["target"]
    
    # Get navigation path based on platform and target type
    navigation_path = get_navigation_path(platform, target["type"])
    
    # Format the implementation guide
    guide = template.format(
        navigation_path=navigation_path,
        target_name=target.get("name", ""),
        target_id=target.get("id", ""),
        campaign_name=target.get("campaign_name", ""),
        campaign_id=target.get("campaign_id", ""),
        current_value=recommendation["current_value"],
        recommended_value=recommendation["recommended_value"]
    )
    
    return guide
```

## Recommendation API

The recommendation engine exposes an API for integration with the Account Health Predictor and other systems:

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/recommendations/generate` | POST | Generate recommendations for an account |
| `/recommendations/{id}` | GET | Retrieve a specific recommendation |
| `/recommendations/account/{account_id}` | GET | Get all recommendations for an account |
| `/recommendations/feedback` | POST | Submit feedback on a recommendation |
| `/recommendations/implement` | POST | Mark a recommendation as implemented |
| `/recommendations/history/{account_id}` | GET | Get historical recommendations |

### Generate Recommendations Request

```json
{
  "account_id": "act_67890",
  "platforms": ["google", "facebook"],
  "timeframe": {
    "start_date": "2025-02-15",
    "end_date": "2025-03-15"
  },
  "health_assessment": {
    "overall_score": 78,
    "issues": [
      {
        "type": "high_cpc",
        "severity": "medium",
        "affected_campaigns": ["camp_12345"]
      },
      {
        "type": "low_quality_score",
        "severity": "high",
        "affected_campaigns": ["camp_23456"]
      }
    ]
  },
  "context": {
    "primary_goal": "efficiency",
    "budget_constraints": true,
    "experimentation_tolerance": "medium"
  },
  "limit": 10
}
```

### Generate Recommendations Response

```json
{
  "request_id": "req_abc123",
  "timestamp": "2025-03-15T14:30:00Z",
  "account_id": "act_67890",
  "recommendation_count": 8,
  "recommendations": [
    {
      "recommendation_id": "rec_12345",
      "category": "bidding",
      "action": "reduce_bids",
      "target": {
        "type": "keywords",
        "id": "kw_34567",
        "name": "premium widgets",
        "campaign_id": "camp_12345",
        "campaign_name": "Widget Campaign"
      },
      "current_value": 1.75,
      "recommended_value": 1.40,
      "change_percentage": -20,
      "rationale": "This keyword has a CPC 40% above average with below-average conversion rate.",
      "expected_impact": {
        "primary": {
          "metric": "cpc",
          "change": "-15% to -25%",
          "confidence": 0.85
        },
        "secondary": {
          "metric": "spend",
          "change": "-10% to -20%",
          "confidence": 0.80
        }
      },
      "priority_score": 85,
      "implementation_guide": "Navigate to Keywords > Widget Campaign > premium widgets and update bid to $1.40"
    },
    // Additional recommendations...
  ]
}
```

## Evaluation Metrics

The recommendation engine's performance is evaluated using the following metrics:

### Effectiveness Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Acceptance Rate | Percentage of recommendations implemented | > 30% |
| Success Rate | Percentage of implemented recommendations that achieved positive impact | > 70% |
| Average Impact | Average performance improvement from implemented recommendations | > 15% |
| Time to Impact | Average days until recommendation shows significant impact | < 14 days |

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Relevance Score | User-rated relevance of recommendations (1-5) | > 4.0 |
| Actionability Score | User-rated actionability of recommendations (1-5) | > 4.2 |
| Clarity Score | User-rated clarity of explanations (1-5) | > 4.3 |
| Accuracy Score | User-rated accuracy of impact predictions (1-5) | > 3.8 |

### System Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Generation Time | Average time to generate recommendations (seconds) | < 5s |
| Diversity Score | Unique recommendation categories per account | > 3 |
| Freshness | Percentage of recommendations based on recent data (< 7 days) | > 90% |

## Future Improvements

Planned enhancements to the recommendation engine include:

### Short-Term Improvements

1. **Multi-Platform Optimization**
   - Enhanced cross-platform recommendation coordination
   - Platform-specific implementation guides

2. **Competitor Analysis Integration**
   - Recommendations based on competitor strategy changes
   - Competitive gap analysis

3. **Advanced Testing Framework**
   - Built-in A/B test design for recommendations
   - Automated recommendation validation

### Long-Term Improvements

1. **Natural Language Processing**
   - Enhanced explanation generation using NLP
   - Conversational interface for recommendation exploration

2. **Reinforcement Learning**
   - Real-time adaptation based on recommendation feedback
   - Multi-step recommendation planning

3. **Causal Inference**
   - Improved impact prediction through causal models
   - Counterfactual reasoning for recommendation validation

4. **Automated Implementation**
   - Direct API integration for one-click implementation
   - Automated scheduling of recommended changes

## References

1. [Account Health Predictor](../account_health_prediction.md)
2. [Model Evaluation Framework](../model_evaluation.md)
3. [Time Series Modeling](./time_series_modeling.md)
4. [Anomaly Detection Methodology](./anomaly_detection.md)

---

*Last updated: March 18, 2025* 