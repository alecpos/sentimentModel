# app/core/feedback/feedback_handler.py
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import uuid4
from app.models.domain.data_catalog_model import DataCatalogModel

class FeedbackProcessor:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def add_feedback(self, catalog_id: str, user_id: str, comment: str) -> Dict[str, Any]:
        """Add feedback to a catalog entry with automated actions"""
        # Use SQLAlchemy 2.0 style query
        stmt = select(DataCatalogModel).filter_by(id=catalog_id)
        entry = self.db_session.scalar(stmt)
        if not entry:
            raise ValueError(f"Catalog entry {catalog_id} not found")

        # Add feedback using the model's method
        feedback_entry = entry.add_feedback(user_id, comment)
        
        # Determine and execute automated actions
        actions = self._determine_automated_actions(entry, comment)
        for action in actions:
            self._execute_automated_action(entry, action)
            
        self.db_session.commit()
        
        return {
            "status": "success",
            "message": "Feedback added successfully",
            "feedback_id": feedback_entry["id"],
            "feedback_count": len(entry.feedback_log or []),
            "automated_actions": actions
        }

    def _determine_automated_actions(self, entry: DataCatalogModel, comment: str) -> List[Dict[str, Any]]:
        """Determine what automated actions to take based on feedback"""
        actions = []
        comment_lower = comment.lower()
        
        # Data freshness related actions
        if any(word in comment_lower for word in ["outdated", "old", "stale"]):
            actions.append({
                "type": "refresh",
                "priority": "high",
                "description": "Schedule data refresh",
                "metadata": {
                    "trigger": "freshness_concern",
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        # Documentation related actions
        if any(word in comment_lower for word in ["unclear", "confusing", "documentation", "docs"]):
            actions.append({
                "type": "documentation",
                "priority": "medium",
                "description": "Flag for documentation review",
                "metadata": {
                    "trigger": "documentation_concern",
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        # Data quality related actions
        if any(word in comment_lower for word in ["incorrect", "wrong", "error", "invalid"]):
            actions.append({
                "type": "quality_check",
                "priority": "high",
                "description": "Trigger data quality validation",
                "metadata": {
                    "trigger": "quality_concern",
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        # Schema related actions
        if any(word in comment_lower for word in ["schema", "structure", "format"]):
            actions.append({
                "type": "schema_review",
                "priority": "medium",
                "description": "Review schema definition",
                "metadata": {
                    "trigger": "schema_concern",
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        return actions

    def _execute_automated_action(self, entry: DataCatalogModel, action: Dict[str, Any]):
        """Execute an automated action based on feedback"""
        # Track action in entry's meta_info
        if 'automated_actions' not in entry.meta_info:
            entry.meta_info['automated_actions'] = []
        
        action_record = {
            "type": action["type"],
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "description": action["description"]
        }
        
        try:
            if action["type"] == "refresh":
                self._handle_refresh_action(entry, action)
                action_record["status"] = "completed"
                
            elif action["type"] == "documentation":
                self._handle_documentation_action(entry, action)
                action_record["status"] = "in_progress"
                
            elif action["type"] == "quality_check":
                self._handle_quality_check_action(entry, action)
                action_record["status"] = "completed"
                
            elif action["type"] == "schema_review":
                self._handle_schema_review_action(entry, action)
                action_record["status"] = "in_progress"
                
        except Exception as e:
            action_record["status"] = "failed"
            action_record["error"] = str(e)
            
        entry.meta_info['automated_actions'].append(action_record)

    def _handle_refresh_action(self, entry: DataCatalogModel, action: Dict[str, Any]):
        """Handle data refresh action"""
        entry.meta_info["refresh_requested"] = True
        entry.meta_info["last_refresh_request"] = datetime.utcnow().isoformat()

    def _handle_documentation_action(self, entry: DataCatalogModel, action: Dict[str, Any]):
        """Handle documentation review action"""
        if "documentation_tasks" not in entry.meta_info:
            entry.meta_info["documentation_tasks"] = []
            
        entry.meta_info["documentation_tasks"].append({
            "type": "review",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "priority": action["priority"]
        })

    def _handle_quality_check_action(self, entry: DataCatalogModel, action: Dict[str, Any]):
        """Handle quality check action"""
        entry.meta_info["quality_check_requested"] = True
        entry.meta_info["last_quality_check_request"] = datetime.utcnow().isoformat()

    def _handle_schema_review_action(self, entry: DataCatalogModel, action: Dict[str, Any]):
        """Handle schema review action"""
        if "schema_reviews" not in entry.meta_info:
            entry.meta_info["schema_reviews"] = []
            
        entry.meta_info["schema_reviews"].append({
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat(),
            "priority": action["priority"]
        })

    def resolve_feedback(self, catalog_id: str, feedback_id: str, resolution: str) -> Dict[str, Any]:
        """Resolve feedback and trigger version update"""
        stmt = select(DataCatalogModel).filter_by(id=catalog_id)
        entry = self.db_session.scalar(stmt)
        if not entry:
            raise ValueError(f"Catalog entry {catalog_id} not found")

        # Use the model's method to resolve feedback
        if entry.resolve_feedback(feedback_id, resolution):
            self.db_session.commit()
            self.db_session.refresh(entry)
            
            return {
                "status": "success",
                "message": "Feedback resolved",
                "new_version": entry.version
            }
        else:
            return {
                "status": "error",
                "message": f"Feedback {feedback_id} not found"
            }

    def get_feedback_stats(self, catalog_id: str) -> Dict[str, Any]:
        """Get feedback statistics for a catalog entry"""
        stmt = select(DataCatalogModel).filter_by(id=catalog_id)
        entry = self.db_session.scalar(stmt)
        if not entry:
            raise ValueError(f"Catalog entry {catalog_id} not found")

        self.db_session.refresh(entry)  # Ensure feedback_log is up to date
        
        feedback_log = entry.feedback_log or []
        total_feedback = len(feedback_log)
        resolved_feedback = len([f for f in feedback_log if f.get("resolved", False)])
        
        return {
            "total_feedback": total_feedback,
            "resolved_feedback": resolved_feedback,
            "pending_feedback": total_feedback - resolved_feedback,
            "resolution_rate": resolved_feedback / total_feedback if total_feedback > 0 else 0,
            "last_feedback": feedback_log[-1] if feedback_log else None,
            "version_history": {
                "current_version": entry.version,
                "last_updated": entry.updated_at.isoformat()
            }
        }

    def get_pending_feedback(self, catalog_id: str = None) -> List[Dict[str, Any]]:
        """Get unresolved feedback for analysis"""
        query = select(DataCatalogModel)
        if catalog_id:
            query = query.filter_by(id=catalog_id)

        pending_feedback = []
        for entry in self.db_session.scalars(query).all():
            self.db_session.refresh(entry)  # Ensure feedback_log is up to date
            feedback_log = entry.feedback_log or []
            for feedback in feedback_log:
                if not feedback.get("resolved", False):
                    pending_feedback.append({
                        "catalog_id": entry.id,
                        "catalog_name": entry.name,
                        "feedback": feedback
                    })

        return pending_feedback

    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """Analyze feedback patterns for system improvement"""
        query = select(DataCatalogModel)
        entries = self.db_session.scalars(query).all()
        
        trends = {
            "total_entries": len(entries),
            "entries_with_feedback": 0,
            "total_feedback": 0,
            "resolution_times": [],
            "common_issues": {},
            "avg_resolution_time": 0
        }

        for entry in entries:
            self.db_session.refresh(entry)  # Ensure feedback_log is up to date
            feedback_log = entry.feedback_log or []
            if feedback_log:
                trends["entries_with_feedback"] += 1
                trends["total_feedback"] += len(feedback_log)

                for feedback in feedback_log:
                    if feedback.get("resolved", False) and "resolved_at" in feedback:
                        created = datetime.fromisoformat(feedback["timestamp"])
                        resolved = datetime.fromisoformat(feedback["resolved_at"])
                        resolution_time = (resolved - created).total_seconds() / 3600  # hours
                        trends["resolution_times"].append(resolution_time)

                    # Track common issues/themes
                    comment = feedback["comment"].lower()
                    for keyword in ["missing", "unclear", "outdated", "error"]:
                        if keyword in comment:
                            trends["common_issues"][keyword] = trends["common_issues"].get(keyword, 0) + 1

        if trends["resolution_times"]:
            trends["avg_resolution_time"] = sum(trends["resolution_times"]) / len(trends["resolution_times"])
        
        return trends