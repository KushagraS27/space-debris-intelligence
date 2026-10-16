# Dataset Card

## Project
Low Cost Uncertainty Aware Space Debris Intelligence

## Version
0.1

## Purpose
Build a reproducible research pipeline for screening catalogued space objects.
This system is for research screening only. It does not produce certified collision warnings or collision probabilities.

## Primary source
CelesTrak public General Perturbations data in OMM compatible JSON format.

## Pilot snapshot
- Initial test group: `STATIONS`
- Purpose: verify download, raw-data versioning, checksum creation, and schema validation.
- Storage: `data/raw/snapshots/<timestamp>/`
- Raw records are not committed to GitHub.

## Research population to add next
- Debris group: `COSMOS-2251-DEBRIS`
- Reference population: selected active satellites.
- Initial orbital regime: low Earth orbit.

## Stored provenance
Each snapshot manifest records:
- source URL
- retrieval time in UTC
- requested CelesTrak group
- format
- record count
- SHA 256 checksum

## Known limitations
- GP/OMM records are not full uncertainty or covariance data.
- Input age affects forecast quality.
- A close approach found by this project is a screening candidate, not a collision probability.
- The catalogue does not represent every debris object.