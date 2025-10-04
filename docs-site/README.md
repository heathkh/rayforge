# Rayforge Documentation

This directory contains the source files for the Rayforge end-user documentation, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Documentation Structure

The documentation is organized into the following sections:

- **Getting Started**: Installation, first-time setup, and quick start guide
- **User Interface**: Main window, canvas tools, 3D preview, and settings
- **Features**: All operations, multi-layer workflow, camera integration, etc.
- **Machine Setup**: Machine profiles, device configuration, and GRBL settings
- **File Formats**: Importing files, supported formats, and exporting G-code
- **Troubleshooting**: Common issues and their solutions
- **Reference**: Keyboard shortcuts, G-code dialects, firmware compatibility
- **Contributing**: Development setup and how to contribute

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
# Using pixi (recommended)
pixi install -e docs

# Or using pip
pip install mkdocs-material mike pillow cairosvg
```

### Local Development

Serve the documentation locally with live reload:

```bash
# Simple command (recommended)
pixi run docs-serve

# Or activate the docs environment first
pixi shell -e docs
mkdocs serve
```

The documentation will be available at http://127.0.0.1:8000/

### Building Static Site

Build the documentation to the `site/` directory:

```bash
# Simple command (recommended)
pixi run docs-build

# Or manually with environment
pixi shell -e docs
mkdocs build
```

## Versioning

Documentation versioning is managed using [mike](https://github.com/jimporter/mike), which creates separate version directories and allows users to switch between documentation for different Rayforge versions.

### Deploying Versions

Deploy documentation for a specific version:

```bash
# Deploy using pixi (simple)
VERSION=dev pixi run docs-deploy

# Or manually with mike
pixi shell -e docs
mike deploy dev development --update-aliases
mike deploy 1.0.0 latest --update-aliases
mike set-default latest
```

### Version Workflow

- **Development (`dev`)**: Built from `main` branch, represents work-in-progress
- **Releases (`X.Y.Z`)**: Built from version tags (e.g., `v1.0.0`)
- **Latest**: Alias that points to the most recent stable release

## Automated Deployment

Documentation is automatically built and deployed via GitHub Actions:

- **Pull Requests**: Documentation is built to verify no errors
- **Main Branch**: Development docs deployed to `dev` version
- **Version Tags**: Release docs deployed with version number and `latest` alias

## Adding Content

### Creating New Pages

1. Create a new Markdown file in the appropriate section directory
2. Add the page to the `nav` section in `mkdocs.yml`
3. Use relative links to reference other pages and images

### Using Screenshots

Screenshots enhance the documentation and help users understand the interface:

1. Take clear, high-resolution screenshots (PNG format preferred)
2. Save to `../docs/` directory (shared with README screenshots)
3. Reference in Markdown: `![Description](../images/screenshot-name.png)`
4. Add descriptive alt text for accessibility

### Markdown Extensions

MkDocs Material supports many useful extensions:

#### Admonitions

```markdown
!!! note "Optional Title"
    This is a note admonition.

!!! tip
    This is a tip.

!!! warning
    This is a warning.

!!! danger
    This is a danger/error message.
```

#### Keyboard Shortcuts

```markdown
Press ++ctrl+c++ to copy.
Press ++ctrl+shift+v++ to paste.
```

#### Code Blocks

```markdown
​```bash
pixi run rayforge
​```

​```python
def hello():
    print("Hello, Rayforge!")
​```
```

#### Tabs

```markdown
=== "Linux"
    Linux instructions here

=== "Windows"
    Windows instructions here
```

#### Tables

```markdown
| Header 1 | Header 2 |
|:---------|:---------|
| Cell 1   | Cell 2   |
```

## Style Guidelines

### Writing Style

- **Audience**: End-users, not developers
- **Tone**: Friendly, clear, and concise
- **Voice**: Use second person ("you") and active voice
- **Examples**: Provide concrete examples and screenshots

### Page Structure

Each page should include:

1. **Title**: Clear, descriptive H1 heading
2. **Introduction**: Brief overview of the page content
3. **Body**: Detailed information with examples
4. **Navigation**: Links to related pages at the end

### Screenshots

- Use consistent window themes (prefer default theme)
- Crop to relevant area
- Annotate if necessary to highlight specific elements
- Keep file sizes reasonable (use PNG with compression)

### Code Examples

- Provide complete, working examples
- Use syntax highlighting
- Include explanatory comments
- Show expected output when relevant

## Documentation Standards

### Links

- Use relative links for internal pages: `[text](../section/page.md)`
- Use descriptive link text (not "click here")
- Verify all links work before committing

### Images

- Alt text is required for all images
- Use descriptive file names (e.g., `machine-settings-device-tab.png`)
- Optimize images for web (compress without losing quality)

### Consistency

- Use consistent terminology throughout
- Follow the same structure across similar pages
- Use the same heading hierarchy (H1 for title, H2 for sections, etc.)

## Testing Documentation

Before submitting changes:

1. **Build locally**: Ensure no build errors
   ```bash
   pixi run -e docs docs-build
   ```

2. **Check links**: Verify all internal links work

3. **Review formatting**: Check that Markdown renders correctly

4. **Test on mobile**: Verify mobile responsiveness (Material theme is responsive by default)

5. **Proofread**: Check spelling and grammar

## Deployment

Documentation is automatically deployed when:

- Changes are pushed to `main` branch (deploys to `dev` version)
- Version tags are pushed (e.g., `v1.0.0`, deploys to version and `latest`)

Manual deployment is not typically necessary, but can be done with:

```bash
# Deploy and push to gh-pages branch
pixi run -e docs docs-deploy
```

## Need Help?

- **MkDocs Material**: https://squidfunk.github.io/mkdocs-material/
- **Mike (versioning)**: https://github.com/jimporter/mike
- **Markdown Guide**: https://www.markdownguide.org/

## Contributing

Documentation improvements are always welcome! Please see [Contributing Guide](contributing/index.md) for details on how to contribute.
