import math

FT = 0.3048
KT = 1852 / 3600
FPM = 196.85  # ft/min per m/s

# positions are metres north and east of the runway centre, on flat ground
RUNWAY = {"length_m": 1200.0, "width_m": 30.0, "heading_deg": 0.0, "elevation_m": 30.0}
THRESHOLD_NORTH_M = -600.0
START_NORTH_M = -560.0
# centre of gravity height when parked, subtracted from heights shown to policies
PARKED_AGL_M = 4.3 * FT
GLIDE_SLOPE_DEG, AIM_PAST_THRESHOLD_M = 3.0, 150.0
ORIGIN = (29.593978, -95.163839)  # the runway centre's latitude and longitude


def earth_radii() -> tuple[float, float]:
    """Return metres per radian of latitude and of longitude at the origin (WGS84)"""
    a, e2 = 6378137.0, 6.69437999014e-3
    sine = 1 - e2 * math.sin(math.radians(ORIGIN[0])) ** 2
    return a * (1 - e2) / sine**1.5, a / math.sqrt(sine) * math.cos(math.radians(ORIGIN[0]))


def geodetic(north: float, east: float) -> tuple[float, float]:
    meridian, parallel = earth_radii()
    return ORIGIN[0] + math.degrees(north / meridian), ORIGIN[1] + math.degrees(east / parallel)


def local(latitude: float, longitude: float) -> tuple[float, float]:
    meridian, parallel = earth_radii()
    return math.radians(latitude - ORIGIN[0]) * meridian, math.radians(longitude - ORIGIN[1]) * parallel


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def frame(north: float, east: float, heading_deg: float) -> tuple[float, float]:
    """Rotate a north/east offset into metres along a heading and to its right"""
    heading = math.radians(heading_deg)
    return north * math.cos(heading) + east * math.sin(heading), -north * math.sin(heading) + east * math.cos(heading)


def runway_frame(north: float, east: float) -> tuple[float, float]:
    return frame(north, east, RUNWAY["heading_deg"])


def glide_error(along: float, agl: float) -> float:
    """Return the height above the glide path, in metres"""
    to_aim = THRESHOLD_NORTH_M + AIM_PAST_THRESHOLD_M - along
    return agl - PARKED_AGL_M - math.tan(math.radians(GLIDE_SLOPE_DEG)) * to_aim


def bearing_to(s: dict, point: dict) -> float:
    return math.degrees(math.atan2(point["east_m"] - s["east_m"], point["north_m"] - s["north_m"]))


def distance(s: dict, point: dict) -> int:
    """Return the distance to a point in metres, rounded to 10 to save tokens"""
    return int(round(math.hypot(point["north_m"] - s["north_m"], point["east_m"] - s["east_m"]), -1))


def vertical_text(metres: float) -> str:
    return "at your height" if abs(metres) < 5 else f"{abs(metres):.0f} m {'above' if metres > 0 else 'below'} you"


def side_of(value: float) -> str:
    return "right" if value > 0 else "left"


def turn_text(degrees: float) -> str:
    return "straight ahead" if abs(degrees) < 3 else f"{abs(degrees):.0f} deg to the {side_of(degrees)}"


def offset_text(right_m: float) -> str:
    return "on it" if abs(right_m) < 1 else f"you are {abs(right_m):.0f} m {side_of(right_m)} of it"
