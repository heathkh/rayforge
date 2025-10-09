# Adding Device Drivers

This guide explains how to add support for new laser cutter/engraver hardware to Rayforge.

## Overview

Rayforge uses a driver architecture to support different types of hardware. Drivers handle communication with the device and translate Rayforge's internal commands into device-specific protocols.

## Driver Types

Rayforge currently supports:

- **GRBL-based devices**: Serial communication with GRBL firmware
- **Custom protocols**: Extensible driver interface for proprietary protocols

## Creating a New Driver

### 1. Understand Your Device

Before creating a driver, gather information about your device:

- Communication protocol (serial, USB, network)
- Command format and syntax
- Configuration parameters
- Motion control specifics
- Power/speed control ranges

### 2. Driver Structure

Drivers are located in the `src/rayforge/drivers/` directory. A typical driver includes:

- Device detection and connection logic
- Command translation
- Status monitoring
- Error handling

### 3. Implementation Steps

1. **Create a new driver file** in `src/rayforge/drivers/`
2. **Implement the driver interface** with required methods
3. **Add device detection** to identify compatible hardware
4. **Test thoroughly** with real hardware
5. **Document** device-specific settings and limitations

## Testing Your Driver

- Test basic connection and disconnection
- Verify motion commands (move, home, etc.)
- Test power/speed control
- Verify emergency stop functionality
- Test with actual cutting/engraving jobs

## Submitting Your Driver

1. Follow the [Development Setup](development.md) guide
2. Create a feature branch for your driver
3. Include documentation for supported devices
4. Add examples and configuration templates
5. Submit a pull request

See the main [Contributing Guide](index.md) for more details on the contribution process.

## Getting Help

If you need help developing a driver:

- Open a [GitHub issue](https://github.com/barebaric/rayforge/issues) with your questions
- Include information about the device you're trying to support
- Provide technical specifications if available
