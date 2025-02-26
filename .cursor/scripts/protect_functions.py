from cursor_utils import Snapshot, SemanticDiff

class FunctionProtector:
    def __init__(self, config_path=".cursor/safeguard-config.json"):
        self.config = load_config(config_path)
        self.snapshot = Snapshot()
        
    def pre_edit_check(self, filepath: str) -> bool:
        snapshot = self.snapshot.create(filepath)
        return self.validate_protection_rules(snapshot)
        
    def post_edit_validate(self, filepath: str) -> bool:
        diff = SemanticDiff(
            original=self.snapshot.latest,
            current=filepath
        )
        return diff.validate_protected_functions() 