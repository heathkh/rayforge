import logging
import numpy as np
import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Adw, Gtk
from blinker import Signal

from .material_test_generator import generate_material_test_ops


logger = logging.getLogger(__name__)


# Define presets for different laser types and operations
PRESETS = {
    "Diode Engrave": {"speed_min": 1000, "speed_max": 5000, "power_min": 10, "power_max": 80, "line_interval": 0.15, "is_engrave": True},
    "CO2 Engrave": {"speed_min": 3000, "speed_max": 15000, "power_min": 10, "power_max": 50, "line_interval": 0.1, "is_engrave": True},
    "Diode Cut": {"speed_min": 100, "speed_max": 500, "power_min": 80, "power_max": 100, "line_interval": 0.5, "is_engrave": False},
    "CO2 Cut": {"speed_min": 200, "speed_max": 1000, "power_min": 30, "power_max": 90, "line_interval": 0.5, "is_engrave": False},
}


class MaterialTestDialog(Adw.Window):
    """A dialog for generating a material test grid."""

    # Signal to emit the generated Ops object
    ops_generated = Signal()

    def __init__(self, parent, doc_editor):
        super().__init__(modal=True, transient_for=parent)
        self.set_title(_("Material Test Generator"))
        self.set_default_size(400, -1)
        self.doc_editor = doc_editor

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.set_content(main_box)

        # --- UI Widgets ---
        self._build_ui(main_box)

        # --- Connect Signals ---
        self.preset_combo.connect("notify::selected-item", self._on_preset_changed)
        self.generate_btn.connect("clicked", self._on_generate)
        self.cancel_btn.connect("clicked", lambda w: self.close())

        # Set initial preset values
        self._on_preset_changed(None, None)

    def _build_ui(self, main_box):
        # Presets Group
        presets_group = Adw.PreferencesGroup(title=_("Presets"))
        main_box.append(presets_group)

        # Combined Preset Dropdown
        self.preset_combo = Gtk.ComboBoxText()
        for preset_name in PRESETS:
            self.preset_combo.append_text(_(preset_name))
        self.preset_combo.set_active(0)
        test_type_row = Adw.ActionRow(title=_("Preset"))
        test_type_row.add_suffix(self.preset_combo)
        presets_group.add(test_type_row)

        # Parameters Group
        params_group = Adw.PreferencesGroup(title=_("Parameters"))
        main_box.append(params_group)

        # Speed Range
        self.speed_min_spin = Gtk.SpinButton.new_with_range(1, 30000, 100)
        self.speed_max_spin = Gtk.SpinButton.new_with_range(1, 30000, 100)
        speed_box = Gtk.Box(spacing=6, orientation=Gtk.Orientation.HORIZONTAL)
        speed_box.append(self.speed_min_spin)
        speed_box.append(Gtk.Label(label="to"))
        speed_box.append(self.speed_max_spin)
        speed_row = Adw.ActionRow(title=_("Speed Range (mm/min)"))
        speed_row.add_suffix(speed_box)
        params_group.add(speed_row)

        # Power Range
        self.power_min_spin = Gtk.SpinButton.new_with_range(1, 100, 1)
        self.power_max_spin = Gtk.SpinButton.new_with_range(1, 100, 1)
        power_box = Gtk.Box(spacing=6, orientation=Gtk.Orientation.HORIZONTAL)
        power_box.append(self.power_min_spin)
        power_box.append(Gtk.Label(label="to"))
        power_box.append(self.power_max_spin)
        power_row = Adw.ActionRow(title=_("Power Range (%)"))
        power_row.add_suffix(power_box)
        params_group.add(power_row)

        # Grid Dimensions
        self.cols_spin = Gtk.SpinButton.new_with_range(2, 20, 1)
        self.rows_spin = Gtk.SpinButton.new_with_range(2, 20, 1)
        grid_box = Gtk.Box(spacing=6, orientation=Gtk.Orientation.HORIZONTAL)
        grid_box.append(self.cols_spin)
        grid_box.append(Gtk.Label(label="x"))
        grid_box.append(self.rows_spin)
        grid_row = Adw.ActionRow(title=_("Grid Dimensions (Speed x Power)"))
        grid_row.add_suffix(grid_box)
        params_group.add(grid_row)

        # Shape Size
        self.shape_size_spin = Gtk.SpinButton.new_with_range(1, 100, 1)
        self.shape_size_spin.set_value(10)
        shape_size_row = Adw.ActionRow(title=_("Shape Size (mm)"))
        shape_size_row.add_suffix(self.shape_size_spin)
        params_group.add(shape_size_row)

        # Spacing
        self.spacing_spin = Gtk.SpinButton.new_with_range(0, 100, 1)
        self.spacing_spin.set_value(5)
        spacing_row = Adw.ActionRow(title=_("Spacing (mm)"))
        spacing_row.add_suffix(self.spacing_spin)
        params_group.add(spacing_row)

        # Line Interval
        self.line_interval_spin = Gtk.SpinButton.new_with_range(0.01, 5.0, 0.01)
        self.line_interval_spin.set_digits(2)
        self.line_interval_row = Adw.ActionRow(title=_("Line Interval (mm)"))
        self.line_interval_row.add_suffix(self.line_interval_spin)
        params_group.add(self.line_interval_row)

        # Options Group
        options_group = Adw.PreferencesGroup(title=_("Options"))
        main_box.append(options_group)

        # Include Labels
        self.labels_check = Gtk.CheckButton(label=_("Include labels for speed and power"))
        self.labels_check.set_active(True)
        labels_row = Adw.ActionRow()
        labels_row.set_activatable_widget(self.labels_check)
        labels_row.set_child(self.labels_check)
        options_group.add(labels_row)

        # Action Buttons
        btn_box = Gtk.Box(spacing=12, orientation=Gtk.Orientation.HORIZONTAL, margin_top=12)
        self.generate_btn = Gtk.Button(label=_("Generate"), css_classes=["suggested-action"])
        self.cancel_btn = Gtk.Button(label=_("Cancel"))
        btn_box.append(self.cancel_btn)
        btn_box.append(self.generate_btn)
        main_box.append(btn_box)

    def _on_preset_changed(self, widget, _):
        """Update spin buttons when a preset is selected."""
        preset_name = self.preset_combo.get_active_text()

        if not preset_name:
            return

        try:
            preset = PRESETS[preset_name]
            self.speed_min_spin.set_value(preset["speed_min"])
            self.speed_max_spin.set_value(preset["speed_max"])
            self.power_min_spin.set_value(preset["power_min"])
            self.power_max_spin.set_value(preset["power_max"])
            self.line_interval_spin.set_value(preset["line_interval"])
            self.line_interval_row.set_sensitive(preset["is_engrave"])
        except KeyError:
            # This might happen if translations don't match keys
            print(f"Warning: Preset not found for {preset_name}")

    def _on_generate(self, widget):
        """Gathers parameters, calls the generator, and emits the result."""
        speed_range = (
            self.speed_min_spin.get_value(),
            self.speed_max_spin.get_value(),
        )
        power_range = (
            self.power_min_spin.get_value(),
            self.power_max_spin.get_value(),
        )
        grid_dimensions = (
            self.cols_spin.get_value_as_int(),
            self.rows_spin.get_value_as_int(),
        )

        preset_name = self.preset_combo.get_active_text()
        test_type = "Engrave" if "Engrave" in preset_name else "Cut"
        laser_type = "Diode" if "Diode" in preset_name else "CO2"

        ops = generate_material_test_ops(
            test_type=test_type,
            laser_type=laser_type,
            speed_range=speed_range,
            power_range=power_range,
            grid_dimensions=grid_dimensions,
            shape_size=self.shape_size_spin.get_value(),
            spacing=self.spacing_spin.get_value(),
            line_interval=self.line_interval_spin.get_value(),
            include_labels=self.labels_check.get_active(),
        )
        self.ops_generated.send(self, ops=ops)
        self.close()

