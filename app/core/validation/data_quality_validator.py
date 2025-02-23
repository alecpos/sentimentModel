from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.domain.data_catalog_model import DataCatalogModel

class DataQualityValidator:
    def __init__(self, db_session: Session):
        self.db = db_session

    def validate_schema(self, catalog_id: str) -> Dict[str, Any]:
        try:
            entry = self.db.query(DataCatalogModel).get(catalog_id)
            if not entry:
                raise ValueError(f"Catalog entry {catalog_id} not found")

            schema_def = entry.schema_definition
            required_fields = schema_def.get("required", [])
            actual_fields = [f["name"] for f in schema_def.get("fields", [])]
            
            missing_fields = [f for f in required_fields if f not in actual_fields]
            
            return {
                "is_valid": len(missing_fields) == 0,
                "missing_fields": missing_fields,
                "field_count": len(actual_fields),
                "required_count": len(required_fields),
                "schema_version": schema_def.get("version", "1.0")
            }
        except Exception as e:
            raise RuntimeError(f"Validation error: {str(e)}")

    def validate_quality(self, catalog_id: str) -> Dict[str, Any]:
        """Validate data quality metrics for a catalog entry"""
        # Use SQLAlchemy 2.0 style query
        stmt = select(DataCatalogModel).filter_by(id=catalog_id)
        entry = self.db.scalar(stmt)
        if not entry:
            raise ValueError(f"Catalog entry {catalog_id} not found")

        # Calculate quality metrics
        quality_metrics = {
            "completeness": self._calculate_completeness(entry),
            "accuracy": self._calculate_accuracy(entry),
            "timeliness": self._calculate_timeliness(entry),
            "consistency": self._check_consistency(entry),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Initialize quality_history if not present
        if 'quality_history' not in entry.meta_info:
            entry.meta_info['quality_history'] = []
            
        # Store quality history
        entry.meta_info['quality_history'].append(quality_metrics)
        
        # Update quality trend if we have history
        if len(entry.meta_info['quality_history']) > 1:
            quality_metrics["trend"] = self._calculate_quality_trend(
                entry.meta_info['quality_history'][-2:]
            )
            
        # Update overall quality score in meta_info
        entry.meta_info['data_quality'] = {
            'current_score': self._calculate_overall_score(quality_metrics),
            'last_checked': quality_metrics['timestamp'],
            'metrics': quality_metrics
        }
            
        self.db.commit()
        return quality_metrics

    def _calculate_completeness(self, entry: DataCatalogModel) -> float:
        """Calculate data completeness score"""
        if not entry.schema_definition or 'fields' not in entry.schema_definition:
            return 0.0
            
        total_fields = len(entry.schema_definition['fields'])
        populated_fields = sum(
            1 for field in entry.schema_definition['fields']
            if field.get('description') and field.get('type')
        )
        
        return round(populated_fields / total_fields, 2) if total_fields > 0 else 0.0

    def _calculate_accuracy(self, entry: DataCatalogModel) -> float:
        """Calculate data accuracy score based on validation rules"""
        if not entry.schema_definition or 'fields' not in entry.schema_definition:
            return 0.0
            
        total_rules = 0
        passed_rules = 0
        
        for field in entry.schema_definition['fields']:
            if 'validation_rules' in field:
                total_rules += len(field['validation_rules'])
                passed_rules += sum(
                    1 for rule in field['validation_rules']
                    if rule.get('status') == 'passed'
                )
        
        return round(passed_rules / total_rules, 2) if total_rules > 0 else 0.0

    def _calculate_timeliness(self, entry: DataCatalogModel) -> float:
        """Calculate data timeliness score"""
        if not entry.updated_at:
            return 0.0
            
        now = datetime.utcnow()
        age_hours = (now - entry.updated_at).total_seconds() / 3600
        
        # Score decreases as age increases (1.0 for fresh data, 0.0 for very old data)
        freshness_threshold = 168  # 7 days in hours
        score = max(0, 1 - (age_hours / freshness_threshold))
        return round(score, 2)

    def _check_consistency(self, entry: DataCatalogModel) -> float:
        """Check data consistency across related entries"""
        consistency_score = 1.0
        
        # Check lineage consistency
        if entry.lineage and entry.lineage.get('source'):
            source_id = entry.lineage['source'].get('id')
            if source_id:
                # Use SQLAlchemy 2.0 style query
                stmt = select(DataCatalogModel).filter_by(id=source_id)
                source_entry = self.db.scalar(stmt)
                if not source_entry:
                    consistency_score *= 0.8  # Penalize for broken lineage
        
        # Check schema consistency
        if entry.schema_definition:
            if not self._validate_schema_structure(entry.schema_definition):
                consistency_score *= 0.9
        
        return round(consistency_score, 2)

    def _calculate_quality_trend(self, history: List[Dict[str, Any]]) -> str:
        """Calculate quality trend from historical data"""
        if len(history) < 2:
            return "stable"
            
        current = self._calculate_overall_score(history[-1])
        previous = self._calculate_overall_score(history[-2])
        
        diff = current - previous
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics"""
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "timeliness": 0.2,
            "consistency": 0.2
        }
        
        score = sum(
            metrics[metric] * weight
            for metric, weight in weights.items()
            if metric in metrics
        )
        
        return round(score, 2)

    def _validate_schema_structure(self, schema: Dict[str, Any]) -> bool:
        """Validate schema structure consistency"""
        required_field_attrs = {'name', 'type', 'description'}
        
        if 'fields' not in schema:
            return False
            
        return all(
            all(attr in field for attr in required_field_attrs)
            for field in schema['fields']
        )