# RSoXS API Prototype - Quick Start

## What's Been Built

A **working prototype** of the high-level beamline control API with the following features:

### ✅ Core Features Implemented

1. **DataFrame-based Scan Interface**
   - Define scans as pandas DataFrames
   - Smart exposure column detection ("exposure", "exp", "Unnamed: 2", etc.)
   - Motor column validation against type literals

2. **Uncertainty Propagation**
   - All AI data returned with `ufloat` (mean ± std_error)
   - Automatic error propagation through calculations
   - Results stored as `_mean` and `_std` columns

3. **Race Condition Prevention**
   - Async context managers for motors and shutters
   - Proper `wait_for_motors()` with status checking
   - Settling time with abort checking
   - Guaranteed resource cleanup

4. **Multi-Motor Support**
   - Move multiple motors simultaneously to DataFrame-defined positions
   - Automatic position restoration on errors
   - Timeout handling

5. **Error Handling**
   - Custom exception hierarchy
   - Motors restore to initial positions on errors
   - Shutters always close (even on exception)
   - Thread-safe abort mechanism for Jupyter

## File Structure

```
src/api_dev/
├── __init__.py          # Package exports
├── types.py             # Exceptions, ScanPoint, ScanResult, type literals
├── validation.py        # DataFrame validation, column detection
├── core.py              # Async primitives (context managers, wait functions)
└── scan.py              # ScanPlan, ScanExecutor

prototype.ipynb          # Working demo notebook
```

## Quick Test

```python
import pandas as pd
from api_dev import ScanPlan, ScanExecutor
from bcs import BCSz

# Connect to beamline
server = BCSz.BCSServer()
await server.connect(addr="localhost", port=5577)

# Define scan as DataFrame
scan_df = pd.DataFrame({
    "Sample X": [10.0, 10.5, 11.0, 11.5, 12.0],
    "Sample Y": [0.0, 0.0, 0.0, 0.0, 0.0],
    "exposure": [1.0, 1.0, 1.5, 1.5, 2.0]
})

# Create and execute scan
plan = ScanPlan.from_dataframe(
    df=scan_df,
    ai_channels=["Photodiode", "TEY signal", "AI 3 Izero"],
    shutter="Light Output"
)

executor = ScanExecutor(server)
results = await executor.execute_scan(plan)

# Results have ufloat columns
print(results["Photodiode_mean"])  # Nominal values
print(results["Photodiode_std"])   # Uncertainties
```

## Key Design Patterns

### 1. Context Managers for Safety

```python
async with motor_move(server, {"Sample X": 10}) as initial_pos:
    async with shutter_control(server):
        # Data collection
        pass
# Motors restored, shutter closed automatically
```

### 2. Abort Mechanism

```python
# In cell 1:
task = asyncio.create_task(executor.execute_scan(plan))

# In cell 2 (while running):
await executor.abort()  # Safe cancellation
```

### 3. Uncertainty as First-Class

```python
from uncertainties import unumpy as unp

# Reconstruct ufloat arrays
signal = unp.uarray(results["Photodiode_mean"], results["Photodiode_std"])
i0 = unp.uarray(results["AI 3 Izero_mean"], results["AI 3 Izero_std"])

# Automatic error propagation
transmission = signal / i0
```

## Testing the Prototype

Open `prototype.ipynb` and run through the tests:

1. **Test 1-2**: DataFrame validation
2. **Test 3-4**: Execute simple scan
3. **Test 5-6**: Analyze results with uncertainty
4. **Test 7**: Test abort mechanism
5. **Test 8**: Multi-motor scan

## What's Next

### Immediate Next Steps
1. ✅ Test prototype with real beamline
2. ⏳ Integrate with existing `RsoxsServer` class
3. ⏳ Add NEXAFS-specific calculations
4. ⏳ Write unit tests

### Future Enhancements
- Feedback-driven scans
- Auto-alignment
- CCD integration
- More sophisticated error recovery

## Notes

- **CCD integration** deferred until core is solid
- **Type safety** fully implemented with type literals
- **Backward compatibility** maintained (old code still works)
- **Low-level primitives** usable directly for custom workflows

## Dependencies

All required packages should already be in your environment:
- pandas
- numpy
- uncertainties
- pydantic
- asyncio (built-in)
- tqdm (optional, for progress bars)

---

**Status**: ✅ Prototype ready for testing!
