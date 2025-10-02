#!/bin/bash
set -e

# --- Setup & Cleanup ---
BUILD_DIR=$(mktemp -d)
ORIG_DIR=$(pwd)
cleanup() {
    echo "--- Cleaning up temporary build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
    echo "Cleanup complete."
}
trap cleanup EXIT

# --- 1. Dynamic Version Detection ---
echo "--- Determining version from Git repository ---"
if [[ -n "$GITHUB_REF_NAME" && "$GITHUB_REF_TYPE" == "tag" ]]; then
    UPSTREAM_VERSION_RAW="$GITHUB_REF_NAME"
else
    UPSTREAM_VERSION_RAW=$(git describe --tags --always --long | sed -e 's/^v//' -e 's/\([^-]*\)-\([0-9]*\)-g\([0-9a-f]*\)/\1~dev\2~\3/')
fi
UPSTREAM_VERSION="${UPSTREAM_VERSION_RAW#v}"
echo "Detected upstream version: ${UPSTREAM_VERSION}"

# --- 2. Vendor Dependencies: Pre-download wheels ---
echo "--- Vendoring pre-built wheels ---"
TMP_SRC_DIR="${BUILD_DIR}/rayforge-${UPSTREAM_VERSION}"
mkdir -p "${TMP_SRC_DIR}/vendor/sdist"

REQUIREMENTS_FILE="debian/requirements-bundle.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "::error::File not found: $REQUIREMENTS_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "::error::jq is not installed. Please install it to continue."
    exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" != *"=="* ]]; then
        echo "::error::Invalid requirement format (must contain '=='): $line"
        exit 1
    fi
    package="${line%%==*}"
    version="${line##*==}"
    if [[ -z "$package" ]] || [[ -z "$version" ]]; then
        echo "::error::Invalid requirement format: $line"
        exit 1
    fi
    echo "Fetching wheel for $package==$version..."
    json_url="https://pypi.org/pypi/${package}/${version}/json"
    response=$(curl -sSL --fail --user-agent "Mozilla/5.0 (compatible; Debian-PPA-Build/1.0)" "$json_url")
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "::error::PyPI returned invalid JSON for $package==$version"
        exit 1
    fi
    source_url=$(echo "$response" | jq -r '.urls[] | select(.packagetype == "sdist") | .url' | head -n 1)
    if [[ -z "$source_url" ]]; then
        echo "::error::No compatible wheel found for $package==$version"
        exit 1
    fi
    filename=$(basename "$source_url")
    echo "Downloading: $filename"
    curl -sSL "$source_url" -o "${TMP_SRC_DIR}/vendor/sdist/${filename}"
done < "$REQUIREMENTS_FILE"

if [[ -z "$(ls -A "${TMP_SRC_DIR}/vendor/sdist/" 2>/dev/null)" ]]; then
    echo "::error::No wheels were downloaded."
    exit 1
fi

# --- 3. Create Upstream Tarball ---
echo "--- Creating upstream tarball with vendored wheels ---"
rsync -a \
    --exclude='.git' \
    --exclude='.pixi' \
    --exclude='.venv' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='repo' \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='debian' \
    "$ORIG_DIR"/ "$TMP_SRC_DIR"/

TARBALL_NAME="rayforge_${UPSTREAM_VERSION}.orig.tar.gz"
tar -czf "$BUILD_DIR/$TARBALL_NAME" -C "$BUILD_DIR" "rayforge-${UPSTREAM_VERSION}"
echo "Created: $BUILD_DIR/$TARBALL_NAME"

# --- 4. Build the Package ---
cd "$BUILD_DIR"
cp -r "$ORIG_DIR/debian" "$TMP_SRC_DIR/"
cd "$TMP_SRC_DIR"

MAINTAINER_INFO=$(grep '^Maintainer:' debian/control | head -n 1 | sed 's/Maintainer: //')
export DEBEMAIL=$(echo "$MAINTAINER_INFO" | sed -E 's/.*<(.*)>.*/\1/')
export DEBFULLNAME=$(echo "$MAINTAINER_INFO" | sed -E 's/ <.*//')

# Set the version string based on whether --source is passed (for PPA) or not (for local testing)
if [[ "${1:-}" == "--source" ]]; then
    # Use the TARGET_DISTRIBUTION from the environment, defaulting to 'noble' if not set
    TARGET_DIST="${TARGET_DISTRIBUTION:-noble}"
    dch --newversion "${UPSTREAM_VERSION}-1~ppa1~${TARGET_DIST}1" --distribution "$TARGET_DIST" "New PPA release for ${TARGET_DIST}."
else
    dch --newversion "${UPSTREAM_VERSION}-1~local1" "New local build ${UPSTREAM_VERSION}."
fi

# Build BOTH source and binary packages at the same time.
# The default command (with no -S or -b) does this.
# The artifacts will be created in the parent directory ($BUILD_DIR)
env -i \
    HOME="$HOME" \
    PATH="/usr/sbin:/usr/bin:/sbin:/bin" \
    DEBEMAIL="$DEBEMAIL" \
    DEBFULLNAME="$DEBFULLNAME" \
    dpkg-buildpackage -us -uc

# --- 5. Copy Artifacts ---
echo "--- Copying build artifacts back to project's dist/ directory ---"
mkdir -p "$ORIG_DIR/dist"
find "$BUILD_DIR" -maxdepth 1 -name 'rayforge*' -type f -exec cp -v {} "$ORIG_DIR/dist/" \;

echo "Build complete. Artifacts are in the dist/ directory."
