"""
RaceChrono Channel ID definitions and utilities

Copyright (C) RaceChrono Oy - All Rights Reserved
Unauthorized copying, distribution and creating derived works of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Antti Lammi in aol@racechrono.com in 2015

Python port - Channel ID constants and manipulation functions.
"""


# Channel ID Bit Structure
"""
  bit
  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
 -----------------------------------------------------------------
 | | |Unused | Sensor    | Sensor  |       Base channel ID       |
 | | |       | Index     | position|                             |
 | |^ Paused channel
  ^ Delta channel                                                |
 \
 -----------------------------------------------------------------

 Sensor position: 0 = no postfix, 1..31 = postfix, see SensorPosition.java
 Sensor index: 0 = no postfix, 1..63 = postfix number
"""


# Channel ID Constants

# General channels
Unknown = 0
Timestamp = 1
DistanceTraveled = 2
PositionLatLong = 3
Speed = 4
Altitude = 5
Bearing = 6
AccelerationLateral = 7
AccelerationLongitudinal = 8
AccelerationX = 9
AccelerationY = 10
AccelerationZ = 11
RateOfRotationX = 12
RateOfRotationY = 13
RateOfRotationZ = 14
UpdateRate = 15
VerticalSpeed = 16
DeviceBatteryTimeToEmpty = 17
DeviceMediaCapacity = 18
DeviceMediaFreeSpace = 19
ElapsedTime = 20
RotationVectorX = 24
RotationVectorY = 25
RotationVectorZ = 26
RotationVectorScalar = 27
MagneticFieldX = 28
MagneticFieldY = 29
MagneticFieldZ = 30
LeanAngle = 37
PitchAngle = 38
YawAngle = 39
HeartRate = 41
AccelerationCombined = 43
Latitude = 44
Longitude = 45
DeviceBatteryLevel = 46
DeviceBatteryVoltage = 47
RollAngle = 48
RollRate = 49
PitchRate = 50
YawRate = 51
RotationX = 52
RotationY = 53
RotationZ = 54
DriftAngle = 55
BarometricPressure = 10062  # Moved from OBD
AmbientAirTemperature = 10031  # Moved from OBD

# Common channels between OBD and data logger: 1000...2000
SteeringAngle = 1001
BrakePedalPosition = 1002
ClutchPedalPosition = 1003
Gear = 1004
GearboxTemperature = 1005
ClutchTemperature = 1006
EngineOilPressure = 1007
EngineOilLevel = 1008
BrakeTemperature = 1009
Odometer = 1010
EngineCoolantPressure = 1011
ThrottleAngle = 1012
TyreTemperature = 1013
TyrePressure = 1014
WheelSpeed = 1015
WheelSpeedFrequency = 1016
SuspensionTravel = 1017
# RearBrakePosition = 1018
EmergencyBrakePosition = 1019
FuelLevelVolume = 1020
EngineCoolantLevel = 1021
EngineCoolantLevelVolume = 1022
AirFuelRatio = 1023
EngineOilLevelVolume = 1024
EngineKnockCorrection = 1025
EngineWastegateDutyCycle = 1026
EnginePower = 1027
EngineCurrent = 1028
EngineTorque = 1029
InverterPower = 1030
BatteryLevelEnergy = 1031
BatteryCurrent = 1032
BrakePressure = 1033
StateOfCharge = 1034
TyreSlip = 1035
BrakeTemperaturePercent = 1036
BatteryTemperature = 1037
BatteryTemperaturePercent = 1038
InverterTemperature = 1039
InverterTemperaturePercent = 1040
CatalyticConverterTemperature = 1041
BatteryLevel = 1042
BatteryVoltage = 1043

EngineRpm = 10024  # Moved from OBD
ThrottlePosition = 10025  # Moved from OBD
EngineCoolantTemperature = 10026  # Moved from OBD
IntakeManifoldPressure = 10027  # Moved from OBD
TimingAdvance = 10028  # Moved from OBD
IntakeAirTemperature = 10029  # Moved from OBD
MafAirflowRate = 10030  # Moved from OBD
EngineLoad = 10032  # Moved from OBD
FuelPressure = 10037  # Moved from OBD
FuelLevel = 10058  # Moved from OBD
EngineOilTemperature = 10066  # Moved from OBD
EngineFuelRate = 10068  # Moved from OBD
AcceleratorPedalPosition = 10071  # Moved from OBD: Accelerator pedal position D
Lambda = 10074  # Moved from OBD
IntakeManifoldRelativePressure = 10075  # Moved from OBD
EthanolFuelContent = 10077  # Moved from OBD

# Generic channels 5000..6000
Analog = 5000
Digital = 5001
Frequency = 5002
Temperature = 5003
Acceleration = 5004
Pressure = 5005
Duration = 5006
DistanceShort = 5007
DistanceMedium = 5008
DistanceLong = 5009
Angle = 5010
Percent = 5011
Voltage = 5012
Weight = 5013
Volume = 5014
ElectricCurrent = 5015
Power = 5016
Torque = 5017
Energy = 5018

# OBD specific channels: 10000..11000
ObdShortTermFuelTrimB1 = 10033
ObdLongTermFuelTrimB1 = 10034
ObdShortTermFuelTrimB2 = 10035
ObdLongTermFuelTrimB2 = 10036
ObdSecondaryAirStatus = 10038
ObdOxygenSensorsPresent = 10039
ObdVehicleStandards = 10040
ObdOxygenSensorsPresent2 = 10041
ObdAuxiliaryInputStatus = 10042
ObdRunTimeSinceStart = 10043
ObdOxygenSensor1B1 = 10044
ObdOxygenSensor2B1 = 10045
ObdOxygenSensor3B1 = 10046
ObdOxygenSensor4B1 = 10047
ObdOxygenSensor1B2 = 10048
ObdOxygenSensor2B2 = 10049
ObdOxygenSensor3B2 = 10050
ObdOxygenSensor4B2 = 10051
ObdDistanceRunWithMIL = 10052
ObdFuelRailPressure = 10053
ObdFuelRailPressureDiesel = 10054
ObdCommanderEGR = 10055
ObdEGRError = 10056
ObdEvaporativePurge = 10057
ObdWarmupsSinceCodesCleared = 10059
ObdDistanceSinceCodesCleared = 10060
ObdEvapSystemPressure = 10061
ObdControlModuleVoltage = 10063
ObdAbsoluteLoadValue = 10064
ObdRelativeThrottlePosition = 10065
ObdRelativeAcceleratorPedalPos = 10067
ObdAbsoluteThrottlePositionB = 10069
ObdAbsoluteThrottlePositionC = 10070
ObdAcceleratorPedalPositionE = 10072
ObdAcceleratorPedalPositionF = 10073

# RC2/RC3 legacy channels 20000..20100
DataLoggerD1Rpm = 20002
DataLoggerAnalog1 = 20003
DataLoggerAnalog2 = 20004
DataLoggerAnalog3 = 20005
DataLoggerAnalog4 = 20006
DataLoggerAnalog5 = 20007
DataLoggerD2 = 20010
DataLoggerAnalog6 = 20011
DataLoggerAnalog7 = 20012
DataLoggerAnalog8 = 20013
DataLoggerAnalog9 = 20014
DataLoggerAnalog10 = 20015
DataLoggerAnalog11 = 20016
DataLoggerAnalog12 = 20017
DataLoggerAnalog13 = 20018
DataLoggerAnalog14 = 20019
DataLoggerAnalog15 = 20020

# GPS 30000..31000
GpsDistanceDirect = 30001
GpsSatellites = 30002
GpsFixType = 30003
GpsHDOP = 30004
GpsVDOP = 30005
GpsPDOP = 30006
GpsAccuracy = 30007
GpsSatellitePNR1 = 30100
# GpsSatellitePNR12 = 30111
GpsSatelliteElevation1 = 30112
# GpsSatelliteElevation12 = 30123
GpsSatelliteAzimuth1 = 30124
# GpsSatelliteAzimuth12 = 30135
GpsSatelliteSNR1 = 30136
# GpsSatelliteSNR12 = 30147
GpsSatelliteLocked1 = 30148
# GpsSatelliteLocked12 = 30159

# Timer 32000..32099
LapComparisonType = 32000  # Not actually used as real channel
LapComparisonTime = 32001
LapComparisonNumber = 32002
LapPreviousTime = 32003
LapPreviousSectorTime = 32004  # Not actually used as real channel
LapPreviousNumber = 32005
LapCurrentStartTime = 32008  # Not actually used as real channel
LapCurrentStartDistance = 32009  # Not actually used as real channel
LapCurrentSectorStartTime = 32010  # Not actually used as real channel
LapCurrentSectorStartDistance = 32011  # Not actually used as real channel
LapCurrentTime = 32012
LapCurrentDistance = 32013
LapCurrentNumber = 32014
LapCurrentSectorTime = 32015  # Not actually used as real channel
LapCurrentSectorDistance = 32016  # Not actually used as real channel
LapRaceStartTime = 32022  # Not actually used as real channel
LapRaceStartDistance = 32023  # Not actually used as real channel
LapTotalRaceTime = 32024
LapTotalRaceDistance = 32025
LapBestTime = 32026
LapBestNumber = 32027
LapPreviousFinishTime = 32028  # Not actually used as real channel
LapPreviousSplitFinishTime = 32029  # Not actually used as real channel
LapCurrentTimeGain = 32030  # Used only as delta channel

# Internal use only
MAX_LIVE_DATA_COUNT = 16
InternalLivePid = 32700
InternalLiveData0 = 32701
InternalLiveDataN = InternalLiveData0 + MAX_LIVE_DATA_COUNT - 1

# Constants
MAX_GPS_SATELLITE_COUNT = 12
MAX_SENSOR_INDEX = 0x3F


# Channel ID name mapping for display
CHANNEL_NAMES = {
    # General channels
    Unknown: "Unknown",
    Timestamp: "Timestamp",
    DistanceTraveled: "DistanceTraveled",
    PositionLatLong: "PositionLatLong",
    Speed: "Speed",
    Altitude: "Altitude",
    Bearing: "Bearing",
    AccelerationLateral: "AccelerationLateral",
    AccelerationLongitudinal: "AccelerationLongitudinal",
    AccelerationX: "AccelerationX",
    AccelerationY: "AccelerationY",
    AccelerationZ: "AccelerationZ",
    RateOfRotationX: "RateOfRotationX",
    RateOfRotationY: "RateOfRotationY",
    RateOfRotationZ: "RateOfRotationZ",
    UpdateRate: "UpdateRate",
    VerticalSpeed: "VerticalSpeed",
    DeviceBatteryTimeToEmpty: "DeviceBatteryTimeToEmpty",
    DeviceMediaCapacity: "DeviceMediaCapacity",
    DeviceMediaFreeSpace: "DeviceMediaFreeSpace",
    ElapsedTime: "ElapsedTime",
    RotationVectorX: "RotationVectorX",
    RotationVectorY: "RotationVectorY",
    RotationVectorZ: "RotationVectorZ",
    RotationVectorScalar: "RotationVectorScalar",
    MagneticFieldX: "MagneticFieldX",
    MagneticFieldY: "MagneticFieldY",
    MagneticFieldZ: "MagneticFieldZ",
    LeanAngle: "LeanAngle",
    PitchAngle: "PitchAngle",
    YawAngle: "YawAngle",
    HeartRate: "HeartRate",
    AccelerationCombined: "AccelerationCombined",
    Latitude: "Latitude",
    Longitude: "Longitude",
    DeviceBatteryLevel: "DeviceBatteryLevel",
    DeviceBatteryVoltage: "DeviceBatteryVoltage",
    RollAngle: "RollAngle",
    RollRate: "RollRate",
    PitchRate: "PitchRate",
    YawRate: "YawRate",
    RotationX: "RotationX",
    RotationY: "RotationY",
    RotationZ: "RotationZ",
    DriftAngle: "DriftAngle",

    # Common channels
    SteeringAngle: "SteeringAngle",
    BrakePedalPosition: "BrakePedalPosition",
    ClutchPedalPosition: "ClutchPedalPosition",
    Gear: "Gear",
    GearboxTemperature: "GearboxTemperature",
    ClutchTemperature: "ClutchTemperature",
    EngineOilPressure: "EngineOilPressure",
    EngineOilLevel: "EngineOilLevel",
    BrakeTemperature: "BrakeTemperature",
    Odometer: "Odometer",
    EngineCoolantPressure: "EngineCoolantPressure",
    ThrottleAngle: "ThrottleAngle",
    TyreTemperature: "TyreTemperature",
    TyrePressure: "TyrePressure",
    WheelSpeed: "WheelSpeed",
    WheelSpeedFrequency: "WheelSpeedFrequency",
    SuspensionTravel: "SuspensionTravel",
    EmergencyBrakePosition: "EmergencyBrakePosition",
    FuelLevelVolume: "FuelLevelVolume",
    EngineCoolantLevel: "EngineCoolantLevel",
    EngineCoolantLevelVolume: "EngineCoolantLevelVolume",
    AirFuelRatio: "AirFuelRatio",
    EngineOilLevelVolume: "EngineOilLevelVolume",
    EngineKnockCorrection: "EngineKnockCorrection",
    EngineWastegateDutyCycle: "EngineWastegateDutyCycle",
    EnginePower: "EnginePower",
    EngineCurrent: "EngineCurrent",
    EngineTorque: "EngineTorque",
    InverterPower: "InverterPower",
    BatteryLevelEnergy: "BatteryLevelEnergy",
    BatteryCurrent: "BatteryCurrent",
    BrakePressure: "BrakePressure",
    StateOfCharge: "StateOfCharge",
    TyreSlip: "TyreSlip",
    BrakeTemperaturePercent: "BrakeTemperaturePercent",
    BatteryTemperature: "BatteryTemperature",
    BatteryTemperaturePercent: "BatteryTemperaturePercent",
    InverterTemperature: "InverterTemperature",
    InverterTemperaturePercent: "InverterTemperaturePercent",
    CatalyticConverterTemperature: "CatalyticConverterTemperature",
    BatteryLevel: "BatteryLevel",
    BatteryVoltage: "BatteryVoltage",

    # Engine/OBD channels
    EngineRpm: "EngineRpm",
    ThrottlePosition: "ThrottlePosition",
    EngineCoolantTemperature: "EngineCoolantTemperature",
    IntakeManifoldPressure: "IntakeManifoldPressure",
    TimingAdvance: "TimingAdvance",
    IntakeAirTemperature: "IntakeAirTemperature",
    MafAirflowRate: "MafAirflowRate",
    AmbientAirTemperature: "AmbientAirTemperature",
    EngineLoad: "EngineLoad",
    FuelPressure: "FuelPressure",
    FuelLevel: "FuelLevel",
    BarometricPressure: "BarometricPressure",
    EngineOilTemperature: "EngineOilTemperature",
    EngineFuelRate: "EngineFuelRate",
    AcceleratorPedalPosition: "AcceleratorPedalPosition",
    Lambda: "Lambda",
    IntakeManifoldRelativePressure: "IntakeManifoldRelativePressure",
    EthanolFuelContent: "EthanolFuelContent",

    # Generic channels
    Analog: "Analog",
    Digital: "Digital",
    Frequency: "Frequency",
    Temperature: "Temperature",
    Acceleration: "Acceleration",
    Pressure: "Pressure",
    Duration: "Duration",
    DistanceShort: "DistanceShort",
    DistanceMedium: "DistanceMedium",
    DistanceLong: "DistanceLong",
    Angle: "Angle",
    Percent: "Percent",
    Voltage: "Voltage",
    Weight: "Weight",
    Volume: "Volume",
    ElectricCurrent: "ElectricCurrent",
    Power: "Power",
    Torque: "Torque",
    Energy: "Energy",

    # GPS channels
    GpsDistanceDirect: "GpsDistanceDirect",
    GpsSatellites: "GpsSatellites",
    GpsFixType: "GpsFixType",
    GpsHDOP: "GpsHDOP",
    GpsVDOP: "GpsVDOP",
    GpsPDOP: "GpsPDOP",
    GpsAccuracy: "GpsAccuracy",

    # Timer/Lap channels
    LapComparisonTime: "LapComparisonTime",
    LapComparisonNumber: "LapComparisonNumber",
    LapPreviousTime: "LapPreviousTime",
    LapPreviousNumber: "LapPreviousNumber",
    LapCurrentTime: "LapCurrentTime",
    LapCurrentDistance: "LapCurrentDistance",
    LapCurrentNumber: "LapCurrentNumber",
    LapTotalRaceTime: "LapTotalRaceTime",
    LapTotalRaceDistance: "LapTotalRaceDistance",
    LapBestTime: "LapBestTime",
    LapBestNumber: "LapBestNumber",
    LapCurrentTimeGain: "LapCurrentTimeGain",
}


# Channel ID Helper Functions

def get_base_channel_id(channel_id: int) -> int:
    """
    Extract base channel ID (lower 15 bits)
    @return channelId without index or position
    """
    return channel_id & 0x7FFF


def get_sensor_index(channel_id: int) -> int:
    """
    Extract sensor index from channel ID
    @return index from channelId (0..63)
    """
    return (channel_id >> 20) & 0x3F


def get_sensor_position(channel_id: int) -> int:
    """
    Extract sensor position from channel ID
    @return position from channelId (0..31)
    """
    return (channel_id >> 15) & 0x1F


def is_delta(channel_id: int) -> bool:
    """
    Check if channel is a delta channel
    @return true if channel is delta channel
    """
    return (channel_id & 0x80000000) != 0


def is_paused(channel_id: int) -> bool:
    """
    Check if channel is a paused channel
    @return true if channel is a paused channel
    """
    return (channel_id & 0x40000000) != 0


def with_index(channel_id: int, sensor_index: int) -> int:
    """
    Add sensor index to channel ID
    @return channelId with index added
    """
    return channel_id | ((sensor_index & 0x3F) << 20)


def without_index(channel_id: int) -> int:
    """
    Remove sensor index from channel ID
    @return channelId without index
    """
    return channel_id & ~(0x3F << 20)


def with_position(channel_id: int, sensor_position: int) -> int:
    """
    Add sensor position to channel ID
    @return channelId with position added
    """
    return channel_id | ((sensor_position & 0x1F) << 15)


def without_position(channel_id: int) -> int:
    """
    Remove sensor position from channel ID
    @return channelId without position
    """
    return channel_id & ~(0x1F << 15)


def with_delta(channel_id: int) -> int:
    """
    Add delta flag to channel ID
    @return channelId with delta
    """
    return channel_id | 0x80000000


def without_delta(channel_id: int) -> int:
    """
    Remove delta flag from channel ID
    @return channelId without delta
    """
    return channel_id & ~0x80000000


def with_paused(channel_id: int) -> int:
    """
    Add paused flag to channel ID
    @return channelId with paused
    """
    return channel_id | 0x40000000


def is_available(channel_id: int, is_comparison: bool, has_comparison: bool) -> bool:
    """
    Get channel availability
    @param channel_id: channel id
    @param is_comparison: true if the channel belongs to a comparison (comparison lap for example)
    @param has_comparison: true if the analysis (or similar) has comparison in place
    @return true if channel should be available
    """
    return (not is_comparison and has_comparison) or not is_delta(channel_id)


def get_importance(channel_id: int) -> int:
    """
    Get importance, lower is more important
    @return importance level (0-5)
    """
    base_id = get_base_channel_id(channel_id)
    if base_id == Unknown:
        return 0
    elif base_id == Speed:
        return 1
    elif base_id == Altitude:
        return 2
    elif base_id == Bearing:
        return 3
    elif base_id == EngineRpm:
        return 4
    else:
        return 5


def is_available_for_custom_channels(channel_id: int) -> bool:
    """
    Check if channel can be used with custom channels
    @return true if channel can be used with custom channels
    """
    base_id = get_base_channel_id(channel_id)
    excluded_channels = {
        Unknown, Timestamp, DistanceTraveled, PositionLatLong, ElapsedTime, UpdateRate,
        GpsDistanceDirect, DataLoggerD1Rpm, DataLoggerAnalog1, DataLoggerAnalog2,
        DataLoggerAnalog3, DataLoggerAnalog4, DataLoggerAnalog5, DataLoggerD2,
        DataLoggerAnalog6, DataLoggerAnalog7, DataLoggerAnalog8, DataLoggerAnalog9,
        DataLoggerAnalog10, DataLoggerAnalog11, DataLoggerAnalog12, DataLoggerAnalog13,
        DataLoggerAnalog14, DataLoggerAnalog15, LapComparisonType, LapComparisonTime,
        LapComparisonNumber, LapPreviousTime, LapPreviousSectorTime, LapPreviousNumber,
        LapCurrentStartTime, LapCurrentStartDistance, LapCurrentSectorStartTime,
        LapCurrentSectorStartDistance, LapCurrentTime, LapCurrentDistance, LapCurrentNumber,
        LapCurrentSectorTime, LapCurrentSectorDistance, LapRaceStartTime, LapRaceStartDistance,
        LapTotalRaceTime, LapTotalRaceDistance, LapBestTime, LapBestNumber,
        LapPreviousFinishTime, LapPreviousSplitFinishTime, LapCurrentTimeGain,
        DeviceBatteryLevel, DeviceBatteryTimeToEmpty, DeviceBatteryVoltage,
        DeviceMediaCapacity, DeviceMediaFreeSpace
    }
    return base_id not in excluded_channels


def is_standard_obd_only(channel_id: int) -> bool:
    """
    Check if channel is only used in standard OBD-II channels
    @return true if channel is only used in standard OBD-II channels
    """
    base_id = get_base_channel_id(channel_id)
    obd_only_channels = {
        ObdShortTermFuelTrimB1, ObdLongTermFuelTrimB1, ObdShortTermFuelTrimB2,
        ObdLongTermFuelTrimB2, ObdSecondaryAirStatus, ObdOxygenSensorsPresent,
        ObdVehicleStandards, ObdOxygenSensorsPresent2, ObdAuxiliaryInputStatus,
        ObdRunTimeSinceStart, ObdOxygenSensor1B1, ObdOxygenSensor2B1, ObdOxygenSensor3B1,
        ObdOxygenSensor4B1, ObdOxygenSensor1B2, ObdOxygenSensor2B2, ObdOxygenSensor3B2,
        ObdOxygenSensor4B2, ObdDistanceRunWithMIL, ObdFuelRailPressure,
        ObdFuelRailPressureDiesel, ObdCommanderEGR, ObdEGRError, ObdEvaporativePurge,
        ObdWarmupsSinceCodesCleared, ObdDistanceSinceCodesCleared, ObdEvapSystemPressure,
        ObdControlModuleVoltage, ObdAbsoluteLoadValue, ObdRelativeThrottlePosition,
        ObdRelativeAcceleratorPedalPos, ObdAbsoluteThrottlePositionB,
        ObdAbsoluteThrottlePositionC, ObdAcceleratorPedalPositionE, ObdAcceleratorPedalPositionF
    }
    return base_id in obd_only_channels


def is_inverted_histogram(channel_id: int) -> bool:
    """
    Check if channel has inverted histogram
    @return true if channel has inverted histogram
    """
    base_id = get_base_channel_id(channel_id)
    if base_id == LapCurrentTime:
        return is_delta(channel_id)
    elif base_id == LapCurrentTimeGain:
        return True
    else:
        return False


def get_channel_name(channel_id: int) -> str:
    """Get human-readable channel name"""
    base_id = get_base_channel_id(channel_id)
    name = CHANNEL_NAMES.get(base_id, f"Channel_{base_id}")

    # Add sensor index/position suffixes if present
    sensor_idx = get_sensor_index(channel_id)
    sensor_pos = get_sensor_position(channel_id)

    if sensor_idx > 0:
        name += f"_{sensor_idx}"
    if sensor_pos > 0:
        name += f"_pos{sensor_pos}"

    return name
