# Flight Simulator

A Cessna 172 on the JSBSim flight dynamics engine, or a quadcopter drone on a simple multirotor model

## Objectives

- **Take off** (`takeoff`): Take off from the runway and climb to 30 m above it at 50 kt or more. The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls below about 48 kt and lands on a runway at 55-80 kt.
- **Fly through rings** (`rings`): Take off, then fly through 4 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. No landing needed. The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls below about 48 kt and lands on a runway at 55-80 kt.
- **Random ring course** (`course`): Take off, then fly through 5 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. The course changes with the seed. No landing needed. The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls below about 48 kt and lands on a runway at 55-80 kt.
- **Ring gauntlet** (`gauntlet`): Take off, then fly through 5 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. They change with the seed: anywhere around the airfield, at very different heights, each facing its own way, so each needs a turn to line up with it. No landing needed. The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls below about 48 kt and lands on a runway at 55-80 kt.
- **Circuit and landing** (`circuit`): Take off, fly the four circuit waypoints in order, make a stable approach and land on the same runway, stopping on it. The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls below about 48 kt and lands on a runway at 55-80 kt.
- **Take off** (`drone-takeoff`): Take off from the runway and climb to 30 m above it. The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies forward at up to 45 kt.
- **Fly through rings** (`drone-rings`): Take off, then fly through 4 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. No landing needed. The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies forward at up to 45 kt.
- **Random ring course** (`drone-course`): Take off, then fly through 5 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. The course changes with the seed. No landing needed. The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies forward at up to 45 kt.
- **Ring gauntlet** (`drone-gauntlet`): Take off, then fly through 5 rings in order. Each ring is a vertical circle of 60 m radius, passed by flying through it in its heading. They change with the seed: anywhere around the airfield, at very different heights, each facing its own way, so each needs a turn to line up with it. No landing needed. The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies forward at up to 45 kt.
- **Circuit and landing** (`drone-circuit`): Take off, fly the four circuit waypoints in order, make a stable approach to the landing spot on the runway centreline, 150 m past its threshold, and land on it. The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies forward at up to 45 kt.

## State

Every decision the model is shown a state with these fields:

- `goal`: the objective, in words
- `situation`: the phase in words: stopped on the runway, rolling down the runway, airborne and climbing out, flying to ring k of n, flying to a waypoint, on final approach, rolling out on the runway, landed, crashed
- `airspeed_kt`: indicated airspeed, knots
- `ground_speed_kt`: speed over the ground, knots
- `height_m`: height above the runway, metres
- `vertical_speed_fpm`: climb (positive) or descent (negative), feet per minute
- `pitch`: the nose's attitude, e.g. 'nose 6 deg up'
- `bank`: 'wings level', or e.g. '15 deg left'
- `heading_deg`: compass heading, degrees
- `stall_warning`: whether the stall warning sounds
- `current_controls`: the answer to each question now in force; for the turn, the heading being held
- `target`: the next point to fly to, relative to the aircraft. On the runway (the Cessna): runway_left_m, centreline and direction. Flying: the ring or waypoint (k of n), distance_m, direction (which way to turn to follow the course: for a ring, its approach line, so as to fly through it in its heading), its height or height band, height_difference, and climb_needed_fpm to arrive at its height. On final approach, the Cessna: runway_threshold_ahead_m, centreline, direction, glide_path and approach_speed_kt; the drone: the landing_spot, distance_m, direction, height_difference and climb_needed_fpm to touch down on it
- `wind`: the direction it blows from and its speed

## Questions

It answers each question with one of its options.

### Cessna 172, on the ground

- `brakes`: Should the pilot hold the wheel brakes now?
  - `off`: released: the aircraft can roll forward and take off
  - `on`: held: the aircraft stops or stays still
- `throttle`: Which throttle setting should the pilot select now?
  - `idle`: no power
  - `half`: cruise power: slower flight turns tighter
  - `full`: maximum power for takeoff and climbing
- `turn`: Which way should the aircraft turn now? Each turn moves the heading it holds by that much.
  - `left 60°`: a big turn left
  - `left 20°`: turn left
  - `left 5°`: a small correction left
  - `straight`: hold the current heading
  - `right 5°`: a small correction right
  - `right 20°`: turn right
  - `right 60°`: a big turn right
- `lift_off`: Should the pilot pull back and lift off now? The aircraft flies from about 55 kt.
  - `not yet`: keep the nosewheel on the runway
  - `lift off`: raise the nose and climb away

### Cessna 172, in the air (flaps on the circuit only)

- `turn`: Which way should the aircraft turn now? Each turn moves the heading it holds by that much.
  - `left 60°`: a big turn left
  - `left 20°`: turn left
  - `left 5°`: a small correction left
  - `straight`: hold the current heading
  - `right 5°`: a small correction right
  - `right 20°`: turn right
  - `right 60°`: a big turn right
- `climb`: Which climb rate or landing flare should the aircraft hold?
  - `climb fast`: +1000 ft/min
  - `climb`: +500 ft/min
  - `level`: hold the current height
  - `flare`: raise the nose to touch down gently at about -100 ft/min
  - `descend slowly`: -250 ft/min
  - `descend`: -500 ft/min
  - `descend fast`: -1000 ft/min
- `speed`: Which airspeed should the autothrottle hold?
  - `idle`: no power: to flare, land or descend steeply
  - `65 kt`: approach and slow flight
  - `80 kt`: climbing
  - `100 kt`: cruise
- `flaps`: Which flap setting should the pilot select now?
  - `up`: takeoff, climb and cruise
  - `10°`: early approach, below 110 kt
  - `20°`: approach, below 85 kt
  - `30°`: final approach and landing, below 85 kt

### Cessna 172, after landing on the circuit

- `brakes`: Should the pilot hold the wheel brakes now to stop on the runway?
  - `off`: released: keep rolling
  - `on`: held: slow down and stop on the runway
- `throttle`: Which throttle setting should the pilot select now to slow down on the runway?
  - `idle`: idle power for the landing rollout
  - `half`: some power, only if needed to keep moving
  - `full`: maximum power for a go-around
- `turn`: Which way should the aircraft turn now? Each turn moves the heading it holds by that much.
  - `left 60°`: a big turn left
  - `left 20°`: turn left
  - `left 5°`: a small correction left
  - `straight`: hold the current heading
  - `right 5°`: a small correction right
  - `right 20°`: turn right
  - `right 60°`: a big turn right

### Quadcopter drone, on the ground and in the air

- `turn`: Which way should the aircraft turn now? Each turn moves the heading it holds by that much.
  - `left 60°`: a big turn left
  - `left 20°`: turn left
  - `left 5°`: a small correction left
  - `straight`: hold the current heading
  - `right 5°`: a small correction right
  - `right 20°`: turn right
  - `right 60°`: a big turn right
- `climb`: Which climb rate or landing flare should the aircraft hold?
  - `climb fast`: +1000 ft/min
  - `climb`: +500 ft/min
  - `level`: hold the current height
  - `flare`: touch down gently at about -100 ft/min
  - `descend slowly`: -250 ft/min
  - `descend`: -500 ft/min
  - `descend fast`: -1000 ft/min
- `speed`: Which ground speed should the drone hold, flying forward?
  - `hover`: stop and hold its position
  - `15 kt`: slow, to approach a spot
  - `45 kt`: full speed

## Timing

Heats run in real time: an answer applies when it arrives and the one before holds meanwhile; after 3 s without one the wings are levelled. In lockstep runs the simulation waits for each decision, then flies 0.5 s. The aircraft's controller holds each answer until the next: a heading, a climb rate and a speed in the air; on the ground the Cessna's brakes, power and lift-off work directly while the nosewheel holds the heading.
