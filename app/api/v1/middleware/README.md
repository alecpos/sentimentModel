# API Middleware Components

This directory contains middleware components for the WITHIN ML Prediction System API.

## Purpose

The middleware system provides capabilities for:
- Handling cross-cutting concerns across API endpoints
- Centralizing common request/response processing logic
- Enforcing security and operational policies
- Monitoring API performance and health
- Standardizing error handling and responses

## Key Components

### Authentication & Authorization

Middleware for security concerns:
- JWT token validation and verification
- API key authentication
- Role-based access control (RBAC)
- Permission checking for specific endpoints
- Session management

### Request Processing

Middleware for handling incoming requests:
- Request validation and sanitization
- Rate limiting and throttling
- Request logging and auditing
- CORS (Cross-Origin Resource Sharing) handling
- Content negotiation

### Response Processing

Middleware for handling outgoing responses:
- Response formatting and standardization
- Compression (gzip, brotli)
- Caching controls
- ETags and conditional responses
- Content-type negotiation

### Monitoring & Diagnostics

Middleware for operational visibility:
- Request timing and performance tracking
- Error rate monitoring
- Resource utilization tracking
- Distributed tracing integration
- Health check endpoint support

## Usage Example

```python
from fastapi import FastAPI
from app.api.v1.middleware import (
    AuthenticationMiddleware, 
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware
)

app = FastAPI()

# Add middleware in the desired order (executed in reverse order)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware, log_request_body=False)
app.add_middleware(
    RateLimitingMiddleware,
    rate_limit=100,  # requests per minute
    by_ip=True
)
app.add_middleware(
    AuthenticationMiddleware,
    exclude_paths=["/health", "/docs", "/openapi.json"]
)

@app.get("/protected-resource")
async def protected_resource(request):
    # This endpoint is now:
    # - Protected by authentication
    # - Rate limited
    # - Logged
    # - Has standardized error handling
    return {"message": "This is a protected resource"}
```

## Integration Points

- **API Routes**: All routes pass through the middleware chain
- **Authentication System**: Middleware verifies credentials with auth system
- **Monitoring**: Metrics from middleware feed into monitoring dashboards
- **Logging**: Request/response details are sent to centralized logging

## Dependencies

- FastAPI middleware system
- Authentication libraries (JWT, OAuth)
- Rate limiting infrastructure
- Metrics collection libraries
- Request/response parsing utilities

## Implementation Details

### Error Handling Middleware

Global error handling middleware that catches exceptions and formats them as standardized error responses:

```python
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """
    Global error handling middleware.
    
    Catches exceptions and formats them as standardized error responses.
    """
    try:
        # Process the request through the endpoint handler
        response = await call_next(request)
        return response
    except HTTPException as e:
        # Handle FastAPI HTTP exceptions
        return JSONResponse(
            status_code=e.status_code,
            content=create_error_response(
                code=get_error_code(e.status_code),
                message=str(e.detail)
            ).dict()
        )
    except ValidationError as e:
        # Handle Pydantic validation errors
        return JSONResponse(
            status_code=400,
            content=create_error_response(
                code="VALIDATION_ERROR",
                message="Invalid request parameters",
                details=format_validation_errors(e)
            ).dict()
        )
    except MLBaseException as e:
        # Handle ML-specific exceptions
        return handle_ml_exception(e)
    except Exception as e:
        # Handle unexpected exceptions
        logger.exception(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred"
            ).dict()
        )
```

### Authentication Middleware

Middleware for handling authentication that extracts and validates JWT tokens, populating request context with authenticated user information:

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Authentication middleware.
    
    Extracts and validates JWT tokens, populating request context with authenticated user.
    """
    # Skip authentication for excluded paths
    if request.url.path in AUTH_EXCLUDED_PATHS:
        return await call_next(request)
    
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content=create_error_response(
                code="UNAUTHORIZED",
                message="Authentication required"
            ).dict()
        )
    
    token = auth_header.replace("Bearer ", "")
    
    try:
        # Validate token and extract user information
        payload = verify_token(token)
        
        # Add user information to request state
        request.state.user = payload
        
        # Process the request
        return await call_next(request)
    except JWTError:
        return JSONResponse(
            status_code=401,
            content=create_error_response(
                code="UNAUTHORIZED",
                message="Invalid or expired token"
            ).dict()
        )
```

### Rate Limiting Middleware

Middleware for enforcing rate limits that tracks request rates per user/IP, enforces configurable rate limits, adds rate limit headers to responses, and rejects requests that exceed limits:

```python
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """
    Rate limiting middleware.
    
    Enforces rate limits based on user ID or IP address.
    """
    # Extract client identifier (user ID or IP)
    client_id = get_client_identifier(request)
    
    # Get endpoint-specific rate limit
    endpoint = request.url.path
    rate_limit = get_rate_limit(endpoint)
    
    # Check rate limit
    if await redis_client.check_rate_limit(client_id, endpoint, rate_limit):
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        limits = await redis_client.get_rate_limit_info(client_id, endpoint)
        response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
        response.headers["X-RateLimit-Remaining"] = str(limits["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limits["reset"])
        
        return response
    else:
        # Rate limit exceeded
        return JSONResponse(
            status_code=429,
            content=create_error_response(
                code="TOO_MANY_REQUESTS",
                message="Rate limit exceeded"
            ).dict()
        )
```

### Request Logging Middleware

Middleware for logging API requests that logs request details (method, path, client), tracks request duration, records response status, and supports configurable log levels:

```python
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Request logging middleware.
    
    Logs request details and timing information.
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request details
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Record start time
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = round((time.time() - start_time) * 1000, 2)
    
    # Log response details
    logger.info(f"Response {request_id}: {response.status_code} ({duration_ms}ms)")
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response
```

### ML Context Middleware

Middleware for setting up ML-specific request context that populates request context with ML environment information, sets up performance monitoring for ML operations, and provides model versioning context:

```python
@app.middleware("http")
async def ml_context_middleware(request: Request, call_next):
    """
    ML context middleware.
    
    Sets up ML-specific request context.
    """
    # Add ML environment information to request state
    request.state.ml_environment = get_ml_environment()
    
    # Set up performance monitoring
    ml_monitor = MLPerformanceMonitor()
    request.state.ml_monitor = ml_monitor
    
    # Process the request
    with ml_monitor:
        response = await call_next(request)
    
    # Add monitoring information to response
    if hasattr(request.state, "ml_metrics"):
        response.headers["X-ML-Processing-Time"] = str(request.state.ml_metrics.processing_time_ms)
    
    return response
```

## Best Practices

When working with middleware:

1. **Order Matters**: Register middleware in the appropriate order (e.g., logging first, error handling early)
2. **Keep It Focused**: Each middleware should handle a single concern
3. **Performance Considerations**: Middleware runs on every request, so keep it efficient
4. **Error Handling**: Handle exceptions within middleware appropriately
5. **Testing**: Write dedicated tests for middleware components
6. **Configuration**: Make middleware behavior configurable

## Adding New Middleware

When adding new middleware:

1. Create a new file in the `middleware` directory
2. Implement the middleware function using the FastAPI middleware pattern
3. Register the middleware in the application setup
4. Add tests in the `tests/api/middleware` directory
5. Document the middleware in this README 