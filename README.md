# Teaching machines to tell SKU numbers apart

## What's in a name?

Electronic parts distributors like Digi-Key, Mouser etc assign their own internal IDs (known as a SKU, or "stock keeping unit") to every product they sell, which is usually different from the "part number" that manufacturers assign to their own products.

For example, `SN74LVC541APWR` is a part number identifying a particular IC made by Texas Instruments. Digi-Key's assigned SKU for it is `296-8521-1-ND`. Mouser calls it `595-SN74LVC541APWR`.

Here are a few more - can you tell which one came from where?
- `2N2222AUA`
- `311-.15LWCT-ND`
- `C0603FR-075K62L`
- `581-12063C475KAT2A`
- `RHM.002AJDKR-ND`

## Teaching the machine to tell them apart

Once you look at a few examples, you'll notice simple patterns that allow you say which string represents which distributor's SKU. If you wanted a computer to do that for you, regular expressions would work. But that wouldn't be fun, would it?

This repository is an attempt to build a machine learning model that can sort part numbers into 3 classes: "manufacturer part number", "Mouser SKU" and "Digi-Key SKU". We are using an [LSTM recurrent neural network](http://adventuresinmachinelearning.com/keras-lstm-tutorial/) implemented in Python using the [Keras](https://keras.io/) framework with [TensorFlow](https://www.tensorflow.org/) backend.

## Results

- [Part 1: Prep the data and train the neural network](parts-distributor-sku-classifier-part-1.ipynb)
- [Part 2: Analyze the results and consider possible improvements to the model](parts-distributor-sku-classifier-part-2-explore.ipynb)
