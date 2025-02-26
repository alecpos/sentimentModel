# Code Complexity Resolution Protocol
**Target File:** {file_path}  
**Current Complexity:** {complexity_score}  
**Acceptable Threshold:** <50  

## Refactoring Steps
1. **Decomposition Plan**  
   - Split `{complex_class}` into:  
     - `{new_class}Base` (Core logic)  
     - `{new_class}Validator` (Validation rules)  
     - `{new_class}Processor` (Data transformations)  

2. **Type Safety Enforcement**  
   - Add type hints to all parameters
   - Implement validation decorators
   - Add return type annotations

3. **Code Organization**
   - Extract complex methods into smaller functions
   - Group related functionality
   - Add clear documentation

4. **Validation Requirements**
   - Maintain existing test coverage
   - Preserve API contracts
   - Keep protected functions (marked with # CursorKeep)

5. **Performance Considerations**
   - Optimize critical paths
   - Consider caching strategies
   - Monitor memory usage

## Implementation Notes
- Use dependency injection where appropriate
- Follow SOLID principles
- Add logging for complex operations
- Include error handling
