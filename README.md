# Library for testing the performance of various convolution algorithms that are part of convolutional neural networks

As part of the project, a framework has been created that can be used by various teams to develop architectures for convolutional neural networks, as a measuring tool for evaluating the performance of operations using various data types and processor instructions.

The framework will provide developers of AI solutions with the following features:
1. Calculation of the performance of convolutional operations in a neural network, including individual layers.
2. Support for various data types, vector processor instructions and basic algorithms.
3. Presets of the most requested neural networks.
4. Ability to test the correctness of the results.
5. Ability to expand to other processors.

**RUNNING LOCALLY**\
**Step 1 – Install dependencies**\
sudo apt install build-essential\
sudo apt install cmake\
sudo apt install git

**Step 2 – Create and change directory**\
mkdir testing\
cd testing

**Step 3 – Clone the project repository and move to the release branch**\
git clone https://github.com/nvdix/CNN_perf.git\
cd CNN_perf\
git checkout release

**Step 4 – Clone json library repository**\
mkdir 3rdparty\
cd 3rdparty\
git clone https://github.com/nlohmann/json.git

**Step 5 – Build the project**\
mkdir build\
cd build\
cmake ..\
make

**Step 6 – The convbench executable file will appear in the build project directory**

**Step 7 - To run the program, you must execute the command**\
./convbench -f=1

**LICENSE**\
The software is available under the [MIT](https://github.com/nvdix/CNN_perf/blob/release/LICENSE.pdf) License.

**CONTACT**\
If you have any questions, feel free to [open an issue](https://github.com/nvdix/CNN_perf/tree/release)
