# SignalFlow

![ci](https://github.com/ideoforms/signal/workflows/ci/badge.svg)

SignalFlow is an audio synthesis framework designed for clear and concise expression of complex musical ideas. It has interfaces for Python and C++, and so can be used for experimentation in iPython/Jupyter or embedded into cross-platform applications.

SignalFlow is in alpha status. All interfaces may be subject to change.

## Example

```python
from signalflow import *

#--------------------------------------------------------------------------------
# An AudioGraph is made up of a network of interconnected Nodes, which generate
# and process audio. 
#--------------------------------------------------------------------------------
graph = AudioGraph()

#--------------------------------------------------------------------------------
# Passing an array of frequencies creates a stereo output.
#--------------------------------------------------------------------------------
sine = Sine([440, 880])

#--------------------------------------------------------------------------------
# Simple attack/sustain/release envelope with linear curves.
#--------------------------------------------------------------------------------
env = EnvelopeASR(0.01, 0.1, 0.5)

#--------------------------------------------------------------------------------
# Use standard arithmetic operations to combine signals.
#--------------------------------------------------------------------------------
output = sine * env

#--------------------------------------------------------------------------------
# Connect the output to the graph, and begin playback.
#--------------------------------------------------------------------------------
graph.add_output(output)
graph.start()
graph.wait()
```

To do the same in C++:

```cpp
#include <signalflow/signalflow.h>
using namespace signalflow;

/*------------------------------------------------------------------------
 * Classes ending in *Ref are memory-managed shared_ptr subclasses.
 *-----------------------------------------------------------------------*/
AudioGraphRef graph = new AudioGraph();

NodeRef sine = new Sine({ 440, 880 });
NodeRef env = new EnvelopeASR(0.01, 0.1, 0.5);
NodeRef ping = sine * env;

graph->add_output(ping);
graph->start();
graph->wait();
```

## Installation

### Dependencies

#### macOS

To install dependencies with Homebrew:

```
brew install libsndfile libsoundio gsl
```

#### Linux, Raspberry Pi

```
apt-get install git cmake g++ libasound2-dev libgsl0-dev libsndfile1-dev libsoundio-dev fftw3-dev
```

### Build (C++)

```
mkdir build
cd build
cmake ..
make -j8
```

### Build (Python)

```
python3 setup.py build
python3 setup.py test
python3 setup.py install
```

## Examples

See [examples](examples) for a number of example programs.

To run an example:
```
cd build
./hello-world
```

## License

For non-commercial use, SignalFlow is available under the terms of the [GPL v3](http://www.gnu.org/licenses/gpl-3.0.en.html).

For commercial use, please [contact the author](http://erase.net/contact).

