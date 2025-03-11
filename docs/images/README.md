# Documentation Images

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This directory contains images used in the WITHIN project documentation. The images are organized into subdirectories based on their purpose and the documentation section they belong to.

## Directory Structure

- `/dashboards` - Images related to dashboard documentation
- `/architecture` - System architecture diagrams
- `/ml` - Machine learning model diagrams and visualizations
- `/api` - API documentation images
- `/guides` - Images for user guides and tutorials
- `/screenshots` - Application screenshots for documentation

## Usage Guidelines

When adding images to the documentation:

1. Place images in the appropriate subdirectory
2. Use descriptive filenames that indicate the content
3. Optimize images for web display (compress when possible)
4. Use PNG format for diagrams and screenshots
5. Use JPG format for photographs
6. Include alt text when referencing images in markdown

## Image Naming Convention

Follow these naming conventions for image files:

- Use lowercase letters and hyphens (no spaces)
- Include relevant component name
- Add descriptive suffix
- Example: `dashboard-performance-overview.png`

## Example Reference

To reference an image in markdown documentation:

```markdown
![Dashboard Performance Overview](/docs/images/dashboards/dashboard-performance-overview.png)
```

## Missing Images

If documentation references an image that does not exist, add a placeholder image with text indicating that the image needs to be created, then update the documentation tracker to indicate that the image is missing. 