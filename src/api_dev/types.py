from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, get_args

import pandas as pd
from uncertainties import ufloat

# ============================================================================
# Custom Exceptions
# ============================================================================


class RsoxsError(Exception):
    """Base exception for RSoXS operations"""

    pass


class MotorError(RsoxsError):
    """Motor operation failed"""

    pass


class MotorTimeoutError(MotorError):
    """Motor move did not complete in time"""

    pass


class ShutterError(RsoxsError):
    """Shutter operation failed"""

    pass


class AcquisitionError(RsoxsError):
    """Data acquisition failed"""

    pass


class ValidationError(RsoxsError):
    """Scan plan validation failed"""

    pass


class ScanAbortedError(RsoxsError):
    """Scan was aborted by user"""

    pass


# ============================================================================
# Type Literals
# ============================================================================

type DIO = Literal[
    "Shutter Rev",
    "Lightfiled Frame Loss",
    "Nothing",
    "Camera Scan",
    "Shutter Output",
    "Air Shutter Output",
    "Light Output",
    "Beam Dumped",
    "PZT Shutter Status",
    "Camera Shutter In",
    "Do Pause Trigger",
    "Trigger Pause Trigger",
    "Shutter Inhibit",
    "Trigger + Inhibit",
]


#  Typing for later
type AI = Literal[
    "TEY signal",
    "Izero",
    "Photodiode",
    "AI 0",
    "AI 3 Izero",
    "AI 5",
    "AI 6 BeamStop",
    "AI 7",
    "Deriv Photodiode",
    "EPU Polarization",
    "Coolstage Temp C",
    "CCD Temperature",
    "Beam Current",
    "Temperature Controller",
    "PZT Shutter",
    "Pause Trigger",
    "LV Memory",
    "Time Stamp Error",
    "Time Stamp Transmit Time",
    "Time Stamp Server Time",
    "Camera Temp Setpoint",
]

type Motor = Literal[
    "Sample Azimuthal Rotation",
    "Piezo Vertical",
    "Piezo Horiz",
    "Sample X",
    "Sample Y",
    "Sample Z",
    "Sample Theta",
    "Sample Y Scaled",
    "CCD Theta",
    "Beam Stop",
    "Pollux CCD X",
    "Pollux CCD Y",
    "CCD X",
    "CCD Y",
    "T-2T",
    "Beamline Energy",
    "Mono 101 Grating",
    "Beamline Energy Goal",
    "Entrance Slit width",
    "Exit Slit Top",
    "Exit Slit Bottom",
    "Exit Slit Left",
    "Exit Slit Right",
    "Horizontal Exit Slit Size",
    "Horizontal Exit Slit Position",
    "Vertical Exit Slit Size",
    "Vertical Exit Slit Position",
    "EPU Gap",
    "EPU Z",
    "Mono Energy",
    "EPU Polarization",
    "M103 Yaw",
    "M103 Bend Up",
    "M103 Bend Down",
    "M101 Feedback",
    "M101 Horizontal Deflection",
    "Entrance Slit Width",
    "M101 Vertical Deflection",
    "Vertical Slit Position",
    "Vertical Slit Size",
    "Horizontal Slit Position",
    "Mono 101 Vessel",
    "Horizontal Slit Size",
    "Diag 106",
    "M121 Translation",
    "PiezoShutter Trans",
    "PZT Shutter",
    "Higher Order Suppressor",
    "AO 0",
    "AO 1",
    "OSP Adjustment",
    "CCD Shutter Control",
    "Temperature Controller",
    "Upstream JJ Vert Aperture",
    "Upstream JJ Vert Trans",
    "Upstream JJ Horz Aperture",
    "Upstream JJ Horz Trans",
    "Middle JJ Vert Aperture",
    "Middle JJ Vert Trans",
    "Middle JJ Horz Aperture",
    "Middle JJ Horz Trans",
    "In-Chamber JJ Vert Aperture",
    "In-Chamber JJ Vert Trans",
    "In-Chamber JJ Horz Aperture",
    "In-Chamber JJ Horz Trans",
    "Sample Number",
    "Coolstage",
    "MCS_axis0",
    "MCS_axis1",
    "MCS_axis2",
    "MCS_axis3",
    "MCS_axis4",
    "Camera Temp Setpoint",
    "CCD Camera Shutter Inhibit",
    "Camera ROI X",
    "Camera ROI Y",
    "Camera ROI Width",
    "Camera ROI Height",
    "Camera ROI X Bin",
    "Camera ROI Y Bin",
    "SampleRot0",
    "SampleRot1",
    "SampleRot2",
    "SampleRot3",
    "SampleRot4",
]

type Command = Literal[
    "None",
    "Normal Move",
    "Backlash Move",
    "Velocity Move",
    "Move to Home",
    "Stop Motor",
    "Set Position",
    "Enable Motor",
    "Disable Motor",
    "Move to Index",
    "Run Home Routine",
    "Set Velocity",
    "Set Acceleration",
    "Set Deceleration",
    "Enable and Move",
    "Disable SW Limits",
    "Enable SW Limits",
    "Start Time Delay",
    "Check Time Delay",
    "Set Output Pulses",
    "Backlash Jog",
    "Normal Jog",
    "Run Coord Program",
    "Halt Coord Program",
    "Gearing ON",
    "Gearing OFF",
    "Set Forward SW Limit",
    "Set Reverse SW Limit",
    "Revert Forward SW Limit",
    "Revert Reverse SW Limit",
]

type Instrument = Literal["CCD", "Photodiode"]

dio = get_args(DIO.__value__)
ai = get_args(AI.__value__)
motor = get_args(Motor.__value__)
command = get_args(Command.__value__)

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ScanPoint:
    """Single point in a scan trajectory"""

    index: int
    motors: Dict[str, float]  # Motor names -> positions
    exposure_time: float
    ai_channels: Optional[List[str]] = None
    delay_after_move: float = 0.2

    def validate(self) -> None:
        """Validate motor names and exposure time"""
        if self.exposure_time <= 0:
            raise ValidationError(f"Invalid exposure time: {self.exposure_time}")

        # Validate motor names
        for motor_name in self.motors.keys():
            if motor_name not in motor:
                raise ValidationError(f"Invalid motor name: {motor_name}")


@dataclass
class ScanResult:
    """Results from a scan point with uncertainty"""

    index: int
    motors: Dict[str, float]
    ai_data: Dict[str, ufloat]  # Channel name -> ufloat value
    exposure_time: float
    timestamp: float
    raw_data: Dict[str, List[float]]  # For debugging

    def to_series(self) -> pd.Series:
        """Convert to pandas Series with proper column names"""
        data = {}
        # Motor positions
        for motor_name, pos in self.motors.items():
            data[f"{motor_name}_position"] = pos
        # AI data with mean and std
        for chan, uval in self.ai_data.items():
            data[f"{chan}_mean"] = uval.nominal_value
            data[f"{chan}_std"] = uval.std_dev
        data["exposure"] = self.exposure_time
        data["timestamp"] = self.timestamp
        return pd.Series(data)
