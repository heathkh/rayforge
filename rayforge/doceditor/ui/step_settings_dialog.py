from typing import Tuple
from gi.repository import Gtk, Adw, GLib, Gdk
from blinker import Signal
from ...config import config
from ...undo import HistoryManager, ChangePropertyCommand
from ...core.doc import Doc
from ...core.step import Step
from ...shared.ui.unit_spin_row import UnitSpinRowHelper
from ...pipeline.transformer import OpsTransformer
from .step_settings import WIDGET_REGISTRY


class StepSettingsDialog(Adw.Window):
    def __init__(self, doc: Doc, step: Step, **kwargs):
        super().__init__(**kwargs)
        self.doc = doc
        self.step = step
        self.history_manager: HistoryManager = doc.history_manager
        self.set_title(_("{name} Settings").format(name=step.name))

        # Used to delay updates from continuous-change widgets like sliders
        # to avoid excessive updates.
        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args: Tuple = ()

        # Safely get machine properties with sensible fallbacks
        if config.machine:
            max_cut_speed = config.machine.max_cut_speed
            max_travel_speed = config.machine.max_travel_speed
        else:
            # Provide sensible defaults if no machine is configured
            max_cut_speed = 3000  # mm/min
            max_travel_speed = 3000  # mm/min

        # Create a vertical box to hold the header bar and the content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Add a header bar for title and window controls (like close)
        header = Adw.HeaderBar()
        main_box.append(header)

        # Set a reasonable default size to avoid being too narrow
        self.set_default_size(600, 750)

        # Destroy window on close to prevent leaks, as a new one is created
        # each time
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)

        # Add a key controller to close the dialog on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # The main content area should be scrollable
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_vexpand(True)  # Allow the scrolled area to grow
        main_box.append(scrolled_window)

        # Create a preferences page and add it to the scrollable area
        page = Adw.PreferencesPage()
        scrolled_window.set_child(page)

        # 1. Producer Settings
        producer_dict = self.step.opsproducer_dict
        if producer_dict:
            producer_name = producer_dict.get("type")
            if producer_name:
                WidgetClass = WIDGET_REGISTRY.get(producer_name)
                if WidgetClass:
                    widget = WidgetClass(
                        title=self.step.typelabel,
                        target_dict=producer_dict,
                        page=page,
                        step=self.step,
                        history_manager=self.history_manager,
                    )
                    page.add(widget)

        # 2. General Settings (check if producer widget wants them shown)
        show_general = True
        if producer_dict:
            producer_name = producer_dict.get("type")
            if producer_name:
                WidgetClass = WIDGET_REGISTRY.get(producer_name)
                if WidgetClass and hasattr(WidgetClass, 'show_general_settings'):
                    show_general = WidgetClass.show_general_settings

        if show_general:
            general_group = Adw.PreferencesGroup(title=_("General Settings"))
            page.add(general_group)

            # Power Slider
            power_row = Adw.ActionRow(title=_("Power (%)"))
            power_adjustment = Gtk.Adjustment(
                upper=100, step_increment=1, page_increment=10
            )
            power_scale = Gtk.Scale(
                orientation=Gtk.Orientation.HORIZONTAL,
                adjustment=power_adjustment,
                digits=0,
                draw_value=True,
            )
            max_power = (
                step.laser_dict.get("max_power", 1000) if step.laser_dict else 1000
            )
            power_percent = (step.power / max_power * 100) if max_power > 0 else 0
            power_adjustment.set_value(power_percent)
            power_scale.set_size_request(300, -1)
            power_scale.connect(
                "value-changed",
                lambda scale: self._debounce(self.on_power_changed, scale),
            )
            power_row.add_suffix(power_scale)
            general_group.add(power_row)

            # Add a spin row for cut speed
            cut_speed_adjustment = Gtk.Adjustment(
                lower=0,
                upper=max_cut_speed,
                step_increment=10,
                page_increment=100,
            )
            cut_speed_row = Adw.SpinRow(
                title=_("Cut Speed"),
                subtitle=_("Max: {max_speed}"),
                adjustment=cut_speed_adjustment,
            )
            self.cut_speed_helper = UnitSpinRowHelper(
                spin_row=cut_speed_row,
                quantity="speed",
                max_value_in_base=max_cut_speed,
            )
            self.cut_speed_helper.set_value_in_base_units(step.cut_speed)
            self.cut_speed_helper.changed.connect(self.on_cut_speed_changed)
            general_group.add(cut_speed_row)

            # Add a spin row for travel speed
            travel_speed_adjustment = Gtk.Adjustment(
                lower=0,
                upper=max_travel_speed,
                step_increment=10,
                page_increment=100,
            )
            travel_speed_row = Adw.SpinRow(
                title=_("Travel Speed"),
                subtitle=_("Max: {max_speed}"),
                adjustment=travel_speed_adjustment,
            )
            self.travel_speed_helper = UnitSpinRowHelper(
                spin_row=travel_speed_row,
                quantity="speed",
                max_value_in_base=max_travel_speed,
            )
            self.travel_speed_helper.set_value_in_base_units(step.travel_speed)
            self.travel_speed_helper.changed.connect(self.on_travel_speed_changed)
            general_group.add(travel_speed_row)

            # Add a switch for air assist
            air_assist_row = Adw.SwitchRow()
            air_assist_row.set_title(_("Air Assist"))
            air_assist_row.set_active(step.air_assist)
            air_assist_row.connect("notify::active", self.on_air_assist_changed)
            general_group.add(air_assist_row)

        # 3. Path Post-Processing Transformers
        if self.step.opstransformers_dicts:
            for t_dict in self.step.opstransformers_dicts:
                transformer_name = t_dict.get("name")
                if transformer_name:
                    WidgetClass = WIDGET_REGISTRY.get(transformer_name)
                    if WidgetClass:
                        transformer = OpsTransformer.from_dict(t_dict)
                        widget = WidgetClass(
                            title=transformer.label,
                            target_dict=t_dict,
                            page=page,
                            step=self.step,
                            history_manager=self.history_manager,
                        )
                        page.add(widget)

        # 4. Post-Step (Assembly) Transformers
        if self.step.post_step_transformers_dicts:
            for t_dict in self.step.post_step_transformers_dicts:
                transformer_name = t_dict.get("name")
                if transformer_name:
                    WidgetClass = WIDGET_REGISTRY.get(transformer_name)
                    if WidgetClass:
                        transformer = OpsTransformer.from_dict(t_dict)
                        widget = WidgetClass(
                            title=transformer.label,
                            target_dict=t_dict,
                            page=page,
                            step=self.step,
                            history_manager=self.history_manager,
                        )
                        page.add(widget)

        self.changed = Signal()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, closing the dialog on Escape or Ctrl+W."""
        has_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        # Gdk.KEY_w covers both lowercase 'w' and uppercase 'W'
        if keyval == Gdk.KEY_Escape or (has_ctrl and keyval == Gdk.KEY_w):
            self.close()
            return True
        return False

    def _on_close_request(self, window):
        # Clean up the debounce timer when the window is closed to prevent
        # a GLib warning.
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        return False  # Allow the window to close

    def _debounce(self, callback, *args):
        """
        Schedules a callback to be executed after a short delay, canceling any
        previously scheduled callback. This prevents excessive updates from
        widgets like sliders.
        """
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)

        self._debounced_callback = callback
        self._debounced_args = args
        # Debounce requests by 150ms
        self._debounce_timer = GLib.timeout_add(
            150, self._commit_debounced_change
        )

    def _commit_debounced_change(self):
        """Executes the debounced callback."""
        if self._debounced_callback:
            self._debounced_callback(*self._debounced_args)

        self._debounce_timer = 0
        self._debounced_callback = None
        self._debounced_args = ()
        return GLib.SOURCE_REMOVE

    def on_power_changed(self, scale):
        max_power = (
            self.step.laser_dict.get("max_power", 1000)
            if self.step.laser_dict
            else 1000
        )
        new_value = max_power / 100 * scale.get_value()
        command = ChangePropertyCommand(
            target=self.step,
            property_name="power",
            new_value=new_value,
            setter_method_name="set_power",
            name=_("Change laser power"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_cut_speed_changed(self, helper: UnitSpinRowHelper):
        new_value = helper.get_value_in_base_units()
        if new_value == self.step.cut_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="cut_speed",
            new_value=new_value,
            setter_method_name="set_cut_speed",
            name=_("Change cut speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_travel_speed_changed(self, helper: UnitSpinRowHelper):
        new_value = helper.get_value_in_base_units()
        if new_value == self.step.travel_speed:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="travel_speed",
            new_value=new_value,
            setter_method_name="set_travel_speed",
            name=_("Change Travel Speed"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)

    def on_air_assist_changed(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.step.air_assist:
            return
        command = ChangePropertyCommand(
            target=self.step,
            property_name="air_assist",
            new_value=new_value,
            setter_method_name="set_air_assist",
            name=_("Toggle air assist"),
        )
        self.history_manager.execute(command)
        self.changed.send(self)
